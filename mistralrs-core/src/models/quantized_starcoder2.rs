#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module};
use mistralrs_quant::QuantMethod;

use crate::attention::PositionEncoding;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    Activation, AttentionConfig, CausalAttention, CausalMasker, MatMul, NonGatedMlp,
    RotaryEmbedding, TransformerBlock,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{
    LanguageModel, LanguageModelExt, Model, TransformContext, TransformerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::loaders::{GgufNaming, GgufWeightSource, TensorNaming, WeightSource};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

/// A transformer block for StarCoder2 using pre-norm architecture.
///
/// Uses the generic `TransformerBlock` with:
/// - `LayerNorm` for normalization
/// - `CausalAttention` for attention (with RoPE)
/// - `NonGatedMlp` for feed-forward (GELU, non-gated)
type StarCoder2Block = TransformerBlock<LayerNorm, CausalAttention, NonGatedMlp>;

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StarCoder2Block>,
    output_norm: LayerNorm,
    output: Arc<dyn QuantMethod>,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub max_seq_len: usize,
    dtype: DType,
}

impl ModelWeights {
    /// Number of transformer layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// starcoder2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f64,
    pub context_window: usize,
    pub rope_freq_base: f32,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("starcoder2")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "feed_forward_length",
            "attention.layer_norm_epsilon",
            "context_length",
        ];
        c.has_required_keys(&required)?;

        Ok(Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            intermediate_size: c.get_value::<u32>("feed_forward_length")? as usize,
            layer_norm_epsilon: c.get_value::<f32>("attention.layer_norm_epsilon")? as f64,
            context_window: c.get_value::<u32>("context_length")? as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(100_000_f32),
        })
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        let metadata = ContentMetadata {
            path_prefix: "starcoder2",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            intermediate_size,
            layer_norm_epsilon,
            context_window,
            rope_freq_base,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for pipeline parallelism
        let layer_range = layer_range.unwrap_or(0..block_count);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(block_count);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < block_count {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start, layer_end, block_count
            );
        }

        let head_dim = embedding_length / head_count;

        // Create weight source and naming
        let mut weights = GgufWeightSource::new(&mut ct);
        let naming = GgufNaming;

        // Load embedding and output weights
        let tok_embeddings = weights.load_embedding(
            &naming.token_embd(),
            0, // vocab_size inferred from tensor
            embedding_length,
            device,
        )?;
        // Get vocab_size from embedding tensor shape (GGUF stores dimensions)
        let vocab_size = tok_embeddings.embeddings().dim(0)?;
        let output_norm = weights.load_layer_norm("output_norm", layer_norm_epsilon, device)?;
        let output = weights.load_linear(&naming.output(), embedding_length, vocab_size, device)?;

        // Create RoPE embeddings per device
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(layer_device.location()) {
                e.insert(Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    context_window,
                    layer_device,
                    true,
                    dtype,
                )?));
            }
        }

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_loaded_layers);

        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&layer_device.location())
                .expect("No RoPE for device location!")
                .clone();

            // Load attention projections with optional biases (Starcoder2 uses biases)
            let q_out_dim = head_count * head_dim;
            let kv_out_dim = head_count_kv * head_dim;
            let q_proj = weights.load_linear_with_optional_bias(&naming.attn_q(layer_idx), embedding_length, q_out_dim, layer_device)?;
            let k_proj = weights.load_linear_with_optional_bias(&naming.attn_k(layer_idx), embedding_length, kv_out_dim, layer_device)?;
            let v_proj = weights.load_linear_with_optional_bias(&naming.attn_v(layer_idx), embedding_length, kv_out_dim, layer_device)?;
            let o_proj = weights.load_linear_with_optional_bias(&naming.attn_output(layer_idx), q_out_dim, embedding_length, layer_device)?;

            // Load MLP weights with optional biases
            let up_proj = weights.load_linear_with_optional_bias(&naming.ffn_up(layer_idx), embedding_length, intermediate_size, layer_device)?;
            let down_proj = weights.load_linear_with_optional_bias(&naming.ffn_down(layer_idx), intermediate_size, embedding_length, layer_device)?;
            let mlp = NonGatedMlp::from_weights(up_proj, down_proj, Activation::GeluPytorchTanh);

            // Load LayerNorms (Starcoder2 uses LayerNorm, not RmsNorm)
            let attn_norm = weights.load_layer_norm(&naming.attn_norm(layer_idx).replace(".weight", ""), layer_norm_epsilon, layer_device)?;
            let ffn_norm = weights.load_layer_norm(&naming.ffn_norm(layer_idx).replace(".weight", ""), layer_norm_epsilon, layer_device)?;

            // Build CausalAttention
            let attn_config = AttentionConfig::new(head_count, head_count_kv, head_dim);
            let mut attention = CausalAttention::new(
                attn_config,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rotary.clone() as Arc<dyn PositionEncoding>,
            )
            .with_attn_dtype(dtype);

            if let AttentionImplementation::PagedAttention = &attention_mechanism {
                attention = attention.with_paged_attn(PagedAttention::new(head_dim, layer_device, None)?);
            }

            layers.push(TransformerBlock::new(attn_norm, attention, ffn_norm, mlp));
        }

        Ok(Self {
            tok_embeddings,
            layers,
            output_norm,
            output,
            mapper: Some(mapper),
            device: device.clone(),
            max_seq_len: context_window,
            dtype,
        })
    }
}

impl ModelWeights {
    fn run_layers(
        &self,
        mut hidden: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        metadata: Option<(&[(Tensor, Tensor)], &PagedAttentionInputMetadata)>,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let layer_metadata = metadata
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta));

            hidden = layer.forward(hidden, mask, position_offsets, &mut cache[i], layer_metadata)?;
        }
        Ok(hidden)
    }
}

// ============================================================================
// Model Trait Implementations
// ============================================================================

impl Model for ModelWeights {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TransformerModel for ModelWeights {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(tokens)
    }

    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let start_offsets: Vec<usize> = vec![ctx.position_offset];

        let mask = CausalMasker.make_causal_mask_as(
            seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        let meta_ref = ctx
            .paged_attn
            .as_ref()
            .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));
        self.run_layers(hidden, mask.as_ref(), &start_offsets, meta_ref, cache)
    }
}

impl LanguageModel for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        let x = self.output_norm.forward(&hidden)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }
}

// Extension trait - accessors and associated types for typed pipelines
impl TransformerModelExt for ModelWeights {
    type Layer = StarCoder2Block;
    type Norm = LayerNorm;

    fn tok_embeddings(&self) -> &Embedding {
        &self.tok_embeddings
    }

    fn layers(&self) -> &[Self::Layer] {
        &self.layers
    }

    fn output_norm(&self) -> &Self::Norm {
        &self.output_norm
    }

    fn mapper(&self) -> Option<&dyn DeviceMapper> {
        self.mapper.as_ref().map(|m| m.as_ref() as &dyn DeviceMapper)
    }

    fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// Extension trait - output accessor for typed pipelines
impl LanguageModelExt for ModelWeights {
    fn output(&self) -> &Arc<dyn QuantMethod> {
        &self.output
    }
}
