#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Phi3 model using the generic transformer infrastructure.
//!
//! Uses `TransformerBlock<RmsNorm, FusedQkvCausalAttention, FusedGatedMlp>` composition
//! to reduce code duplication while handling Phi3's fused weight layout.
//!
//! Supports loading from both GGUF and safetensors formats via `FromGGUF` and
//! `FromSafetensors` traits.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};

use crate::attention::{AttentionConfig, FusedQkvCausalAttention, PositionEncoding};
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    embedding, Activation, CausalMasker, FusedGatedMlp, MatMul, PhiRopeScalingConfig, RmsNorm,
    RotaryEmbedding, TransformerBlock,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{
    LanguageModel, LanguageModelExt, Model, TransformContext, TransformerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::loaders::{GgufWeightSource, WeightSource};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

/// A transformer block for Phi3 using fused QKV attention and fused gated MLP.
///
/// Uses the generic `TransformerBlock` with:
/// - `RmsNorm` for normalization
/// - `FusedQkvCausalAttention` for attention (fused Q+K+V projection)
/// - `FusedGatedMlp` for feed-forward (fused gate+up projection)
type Phi3Block = TransformerBlock<RmsNorm, FusedQkvCausalAttention, FusedGatedMlp>;

/// Phi3 model weights implementing `LanguageModel` trait.
///
/// This is the canonical Phi3 implementation that can be loaded from either:
/// - **GGUF format**: via `FromGGUF` trait (llama.cpp quantized models)
/// - **Safetensors format**: via `FromSafetensors` trait (HuggingFace models)
///
/// The model is stateless - the KV cache is passed into `transform()` from the pipeline.
///
/// # Features
/// - Pipeline parallelism support (layer range loading)
/// - Paged attention support
/// - Device mapping for multi-GPU
/// - Sliding window attention
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Phi3Block>,
    output_norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub max_seq_len: usize,
    dtype: DType,
    n_heads: usize,
}

impl ModelWeights {
    /// Number of transformer layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// phi3 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub intermediate_size: usize,
    pub rope_dim: usize,
    pub rms_eps: f64,
    pub context_window: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("phi3")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "feed_forward_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
            "context_length",
        ];
        c.has_required_keys(&required)?;

        Ok(Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            intermediate_size: c.get_value::<u32>("feed_forward_length")? as usize,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            rms_eps: c.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            context_window: c.get_value::<u32>("context_length")? as usize,
        })
    }
}

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, false);

/// Configuration for Phi3 model loaded from safetensors.
/// Mirrors the config.json structure from HuggingFace.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    pub partial_rotary_factor: Option<f64>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl crate::models::LanguageModelConfig for Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta as f32
    }
    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }
    fn hidden_act(&self) -> crate::layers::Activation {
        self.hidden_act
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn quantization_config(&self) -> Option<&mistralrs_quant::QuantizedConfig> {
        self.quantization_config.as_ref()
    }
}

// =============================================================================
// FromSafetensors Implementation
// =============================================================================

impl ModelConfig::FromSafetensors for ModelWeights {
    type Config = Config;

    fn from_safetensors(
        cfg: &Self::Config,
        vb: ShardedVarBuilder,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }

        let head_dim = cfg.head_dim();
        let num_layers = cfg.num_hidden_layers;

        // Determine layer range for pipeline parallelism
        let layer_range = layer_range.unwrap_or(0..num_layers);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(num_layers);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < num_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                num_layers
            );
        }

        let vb_m = vb.pp("model");

        // Load embeddings
        let tok_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &cfg.quantization_config,
        )?;

        // Create RotaryEmbedding per device
        // Note: Using simple RoPE. For complex scaling (YaRN, etc.), extend this.
        let rope_dim = head_dim; // Phi3 uses full head_dim for RoPE
        let max_seq_len = cfg.sliding_window.unwrap_or(cfg.max_position_embeddings);
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    rope_dim,
                    max_seq_len,
                    layer_device,
                    true, // is_gpt_neox style
                    dtype,
                )?));
            }
        }

        // Load transformer layers
        let vb_l = vb_m.pp("layers");
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

            let vb_layer = vb_l.pp(layer_idx);

            // Load fused QKV projection
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads * head_dim;
            let qkv_proj = mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                op_size,
                &cfg.quantization_config,
                vb_layer.pp("self_attn").pp("qkv_proj"),
            )?;

            // Load output projection
            let o_proj = mistralrs_quant::linear_no_bias(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                vb_layer.pp("self_attn").pp("o_proj"),
            )?;

            // Load fused gate+up projection
            let gate_up_proj = mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                &cfg.quantization_config,
                vb_layer.pp("mlp").pp("gate_up_proj"),
            )?;

            // Load down projection
            let down_proj = mistralrs_quant::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                vb_layer.pp("mlp").pp("down_proj"),
            )?;

            // Load normalization layers
            let attn_norm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("input_layernorm"),
            )?;
            let ffn_norm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("post_attention_layernorm"),
            )?;

            // Build attention with sliding window
            let attn_config =
                AttentionConfig::new(cfg.num_attention_heads, cfg.num_key_value_heads, head_dim)
                    .with_sliding_window(cfg.sliding_window.unwrap_or(cfg.max_position_embeddings));

            let mut attention = FusedQkvCausalAttention::new(
                attn_config,
                qkv_proj,
                o_proj,
                rotary.clone() as Arc<dyn PositionEncoding>,
            )
            .with_attn_dtype(dtype);

            if let AttentionImplementation::PagedAttention = &attention_mechanism {
                attention =
                    attention.with_paged_attn(PagedAttention::new(head_dim, layer_device, None)?);
            }

            // Build MLP with fused gate+up
            let mlp =
                FusedGatedMlp::from_weights(gate_up_proj, down_proj, cfg.hidden_act, cfg.intermediate_size);

            layers.push(TransformerBlock::new(attn_norm, attention, ffn_norm, mlp));
        }

        // Load output norm
        let output_norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Load output projection (may be tied to embeddings)
        let output: Arc<dyn QuantMethod> = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                vb.pp("lm_head"),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                tok_embeddings.embeddings().clone(),
                None,
            ))?
        };

        Ok(Self {
            tok_embeddings,
            layers,
            output_norm,
            output,
            mapper: Some(mapper),
            device: device.clone(),
            max_seq_len: cfg.sliding_window.unwrap_or(cfg.max_position_embeddings),
            dtype,
            n_heads: cfg.num_attention_heads,
        })
    }
}

// =============================================================================
// FromGGUF Implementation
// =============================================================================

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        // Extract configuration from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: "phi3",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            intermediate_size,
            rope_dim,
            rms_eps,
            context_window,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for pipeline parallelism
        let layer_range = layer_range.unwrap_or(0..block_count);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(block_count);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < block_count {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                block_count
            );
        }

        let head_dim = embedding_length / head_count;

        // Create weight source
        let mut weights = GgufWeightSource::new(&mut ct);

        // Load embedding and output weights
        let tok_embeddings =
            weights.load_embedding("token_embd.weight", 0, embedding_length, device)?;
        // Get vocab_size from embedding tensor shape (GGUF stores dimensions)
        let vocab_size = tok_embeddings.embeddings().dim(0)?;
        let output_norm = weights.load_rms_norm("output_norm.weight", rms_eps, device)?;
        let output = weights.load_linear("output.weight", embedding_length, vocab_size, device)?;

        // Create RoPE embeddings per device (reused across layers on same device)
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    10_000.0, // Phi3 uses standard RoPE base
                    rope_dim,
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

            // Load fused QKV and output projections
            // QKV: embedding_length -> (head_count + 2*head_count_kv) * head_dim
            let qkv_out_dim = head_count * head_dim + 2 * head_count_kv * head_dim;
            let qkv_proj = weights.load_linear(
                &format!("blk.{layer_idx}.attn_qkv.weight"),
                embedding_length,
                qkv_out_dim,
                layer_device,
            )?;
            let o_proj = weights.load_linear(
                &format!("blk.{layer_idx}.attn_output.weight"),
                head_count * head_dim,
                embedding_length,
                layer_device,
            )?;

            // Load fused gate+up and down projections
            // Gate+Up is fused: embedding_length -> 2 * intermediate_size
            let gate_up = weights.load_linear(
                &format!("blk.{layer_idx}.ffn_up.weight"),
                embedding_length,
                2 * intermediate_size,
                layer_device,
            )?;
            let down = weights.load_linear(
                &format!("blk.{layer_idx}.ffn_down.weight"),
                intermediate_size,
                embedding_length,
                layer_device,
            )?;

            // Load normalization layers
            let attn_norm = weights.load_rms_norm(
                &format!("blk.{layer_idx}.attn_norm.weight"),
                rms_eps,
                layer_device,
            )?;
            let ffn_norm = weights.load_rms_norm(
                &format!("blk.{layer_idx}.ffn_norm.weight"),
                rms_eps,
                layer_device,
            )?;

            // Build attention with sliding window
            let attn_config = AttentionConfig::new(head_count, head_count_kv, head_dim)
                .with_sliding_window(context_window);

            let mut attention = FusedQkvCausalAttention::new(
                attn_config,
                qkv_proj,
                o_proj,
                rotary.clone() as Arc<dyn PositionEncoding>,
            )
            .with_attn_dtype(dtype);

            if let AttentionImplementation::PagedAttention = &attention_mechanism {
                attention = attention.with_paged_attn(PagedAttention::new(head_dim, layer_device, None)?);
            }

            // Build MLP with fused gate+up
            let mlp = FusedGatedMlp::from_weights(gate_up, down, Activation::Silu, intermediate_size);

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
            n_heads: head_count,
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

    fn transform(
        &self,
        hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        // Create 2D shape tensor for mask function [batch, seq_len]
        let mask_shape = hidden.i((.., .., 0usize))?;
        let start_offsets: Vec<usize> = vec![ctx.position_offset];

        // Determine past KV len source for masking
        let past_kv_len_cache: &dyn PastKvLenCache = if ctx.paged_attn.is_some() {
            &start_offsets.as_slice()
        } else {
            &cache
        };

        // Create sliding window causal mask
        let mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            &mask_shape,
            past_kv_len_cache,
            Some(self.max_seq_len),
            self.dtype,
            self.n_heads,
        )?;

        // Skip mask for non-first chunks in paged attention
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
    type Layer = Phi3Block;
    type Norm = RmsNorm;

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
