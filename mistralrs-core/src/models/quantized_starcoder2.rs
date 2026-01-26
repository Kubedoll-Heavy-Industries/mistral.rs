#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    Activation, AttentionConfig, CausalAttention, CausalMasker, MatMul, NonGatedMlp,
    RotaryEmbedding, TransformerBlock,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{LanguageModel, Model, TransformContext, TransformerModel};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
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

fn layer_norm(w: QTensor, b: QTensor, eps: f64) -> Result<LayerNorm> {
    let w = w.dequantize(&w.device())?;
    let b = b.dequantize(&b.device())?;
    Ok(LayerNorm::new(w, b, eps))
}

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
            "attention.layer_norm_epsilon",
            "context_length",
        ];
        c.has_required_keys(&required)?;

        Ok(Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
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
            layer_norm_epsilon,
            context_window,
            rope_freq_base,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for partial loading (pipeline parallelism)
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

        let tok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let head_dim = embedding_length / head_count;

        let output_norm = layer_norm(
            ct.tensor("output_norm.weight", device)?,
            ct.tensor("output_norm.bias", device)?,
            layer_norm_epsilon,
        )?;
        let output = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(ct.tensor("output.weight", device)?),
            b: None,
        })?);

        // Create RoPE embeddings for loaded layers
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    context_window,
                    device,
                    true,
                    dtype,
                )?),
            );
        }

        let mut layers = Vec::with_capacity(num_loaded_layers);

        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            // Helper to load optional bias and dequantize
            let load_bias = |ct: &mut Content<'_, R>, name: &str| -> Result<Option<Tensor>> {
                match ct.tensor(name, device) {
                    Ok(qt) => Ok(Some(qt.dequantize(device)?)),
                    Err(_) => Ok(None),
                }
            };

            // Attention projections
            let q_proj: Arc<dyn QuantMethod> = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?),
                b: load_bias(&mut ct, &format!("{prefix}.attn_q.bias"))?,
            })?);
            let k_proj: Arc<dyn QuantMethod> = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?),
                b: load_bias(&mut ct, &format!("{prefix}.attn_k.bias"))?,
            })?);
            let v_proj: Arc<dyn QuantMethod> = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?),
                b: load_bias(&mut ct, &format!("{prefix}.attn_v.bias"))?,
            })?);
            let o_proj: Arc<dyn QuantMethod> = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?),
                b: load_bias(&mut ct, &format!("{prefix}.attn_output.bias"))?,
            })?);

            // MLP weights
            let mlp = NonGatedMlp::from_weights(
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?),
                    b: load_bias(&mut ct, &format!("{prefix}.ffn_up.bias"))?,
                })?),
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?),
                    b: load_bias(&mut ct, &format!("{prefix}.ffn_down.bias"))?,
                })?),
                Activation::GeluPytorchTanh,
            );

            // Layer norms
            let attn_norm = layer_norm(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                ct.tensor(&format!("{prefix}.attn_norm.bias"), device)?,
                layer_norm_epsilon,
            )?;
            let ffn_norm = layer_norm(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                ct.tensor(&format!("{prefix}.ffn_norm.bias"), device)?,
                layer_norm_epsilon,
            )?;

            // Build CausalAttention
            let attn_config = AttentionConfig::new(head_count, head_count_kv, head_dim);
            let mut attention = CausalAttention::new(
                attn_config,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rotary.clone() as Arc<dyn crate::attention::PositionEncoding>,
            )
            .with_attn_dtype(dtype);

            if let AttentionImplementation::PagedAttention = &attention_mechanism {
                attention = attention.with_paged_attn(PagedAttention::new(head_dim, device, None)?);
            }

            layers.push(TransformerBlock::new(attn_norm, attention, ffn_norm, mlp));
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
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
