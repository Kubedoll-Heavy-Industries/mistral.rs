#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen2 model using the generic transformer infrastructure.
//!
//! Uses `StandardTransformerBlock` (RmsNorm + CausalAttention + Mlp) composition
//! with optional Q/K normalization support (for Qwen3 compatibility).
//!
//! Supports loading from both GGUF and safetensors formats via `FromGGUF` and
//! `FromSafetensors` traits.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};

use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{Activation, RmsNorm, RmsNormQkNorm};
use crate::models::{
    standard_embed, standard_lm_head, standard_transform, LanguageModel, LanguageModelExt, Model,
    TransformContext, TokenizerModel, TransformerModelExt,
};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::{
    load_transformer_from_safetensors, load_transformer_layers, GgufNaming, GgufWeightSource,
    SafetensorsNaming, StandardTransformerBlock, TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, false);

/// Configuration for Qwen2 model loaded from safetensors.
/// Mirrors the config.json structure from HuggingFace.
#[derive(Debug, Clone, serde::Deserialize, Default, serde::Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
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
// Model Weights
// =============================================================================

/// Qwen2 model weights implementing `LanguageModel` trait.
///
/// This is the canonical Qwen2 implementation that can be loaded from either:
/// - **GGUF format**: via `FromGGUF` trait (llama.cpp quantized models)
/// - **Safetensors format**: via `FromSafetensors` trait (HuggingFace models)
///
/// The model uses pre-norm architecture with:
/// - RMS normalization for attention and FFN
/// - Optional Q/K normalization (when present, e.g., Qwen3 models loaded via Qwen2 path)
/// - Optional Q/K/V attention biases (Qwen2-specific)
/// - Gated MLP with SiLU activation
/// - RoPE positional embeddings
///
/// The model is stateless - the KV cache is passed into `transform()` from the pipeline.
///
/// # Features
/// - Pipeline parallelism support (layer range loading)
/// - Paged attention support
/// - Device mapping for multi-GPU
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    kv_dim: usize,
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
        adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        let naming = SafetensorsNaming;

        // Load transformer with attention bias config and optional Q/K norm
        let loaded = load_transformer_from_safetensors(
            cfg,
            TransformerConfig::from_config(cfg).with_attention_bias(), // Qwen2-specific
            vb,
            device,
            &*mapper,
            attention_mechanism,
            dtype,
            layer_range,
            adapter_registry,
            |ctx, builder, weights| {
                // Check if Q/K norm exists (Qwen3-style) and load if present
                let q_norm_name = naming.attn_q_norm(ctx.layer_idx);
                let k_norm_name = naming.attn_k_norm(ctx.layer_idx);

                if weights.has_tensor(&q_norm_name) && weights.has_tensor(&k_norm_name) {
                    let q_norm =
                        weights.load_rms_norm(&q_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    let k_norm =
                        weights.load_rms_norm(&k_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    Ok(builder.with_qk_norm(Arc::new(RmsNormQkNorm::new(q_norm, k_norm))))
                } else {
                    // No Q/K norm (standard Qwen2)
                    Ok(builder)
                }
            },
        )?;

        Ok(Self {
            tok_embeddings: loaded.tok_embeddings,
            layers: loaded.layers,
            norm: loaded.output_norm,
            output: loaded.output,
            device: device.clone(),
            max_seq_len: loaded.max_seq_len,
            mapper: Some(mapper),
            dtype,
            kv_dim: cfg.head_dim() * cfg.num_key_value_heads,
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
        adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        // Verify architecture (accept both qwen2 and qwen3)
        let meta = ct.get_metadata();
        let arch: String = {
            use crate::utils::gguf_metadata::TryValueInto;
            meta.get("general.architecture").cloned().try_value_into()?
        };
        if arch != "qwen2" && arch != "qwen3" {
            candle_core::bail!("Expected `qwen2` or `qwen3` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata using generic infrastructure
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            // Qwen2 has optional Q/K/V biases (but not O bias)
            .with_attention_bias();

        // Create weight source wrapper
        let mut weights = GgufWeightSource::new(&mut ct);
        let naming = GgufNaming;

        // Load embedding weights
        let tok_embeddings = weights.load_embedding(
            &naming.token_embd(),
            config.vocab_size,
            config.hidden_size,
            device,
        )?;

        // Load output norm
        let norm = weights.load_rms_norm(&naming.output_norm(), config.rms_norm_eps, device)?;

        // Load output weights (tie to embeddings if not present)
        let output = if weights.has_tensor(&naming.output()) {
            weights.load_linear(
                &naming.output(),
                config.hidden_size,
                config.vocab_size,
                device,
            )?
        } else {
            weights.load_linear(
                &naming.token_embd(),
                config.hidden_size,
                config.vocab_size,
                device,
            )?
        };

        // Load transformer layers using generic infrastructure
        // Qwen2 may have optional Q/K norm (when loading qwen3 files through qwen2 path)
        let layers = load_transformer_layers(
            &config,
            &mut weights,
            &naming,
            layer_range,
            &*mapper,
            device,
            attention_mechanism,
            dtype,
            adapter_registry,
            |ctx, builder, weights| {
                // Check if Q/K norm exists (Qwen3-style) and load if present
                let q_norm_name = naming.attn_q_norm(ctx.layer_idx);
                let k_norm_name = naming.attn_k_norm(ctx.layer_idx);

                if weights.has_tensor(&q_norm_name) && weights.has_tensor(&k_norm_name) {
                    let q_norm =
                        weights.load_rms_norm(&q_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    let k_norm =
                        weights.load_rms_norm(&k_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    let qk_norm: Arc<dyn crate::attention::QkNorm> =
                        Arc::new(RmsNormQkNorm::new(q_norm, k_norm));
                    Ok(builder.with_qk_norm(qk_norm))
                } else {
                    // No Q/K norm (standard Qwen2)
                    Ok(builder)
                }
            },
        )?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            mapper: Some(mapper),
            dtype,
            kv_dim: config.head_dim * config.num_kv_heads,
        })
    }
}

impl ModelWeights {
    /// Forward pass for embeddings - returns hidden states before LM head.
    /// Used by GGUF embedding pipeline.
    pub fn forward_hidden_states(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let embeds = self.embed(x)?;
        let ctx = TransformContext {
            seq_len: embeds.dim(1)?,
            position_offset: start_offsets.first().copied().unwrap_or(0),
            paged_attn: None,
            flash_params: None,
            position_ids: None,
        };
        let hidden = self.transform(embeds, &ctx, cache)?;
        // Return hidden states after final norm (before output projection)
        self.output_norm().forward(&hidden)
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

// Object-safe base trait - required methods
impl TokenizerModel<[KvCache]> for ModelWeights {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        standard_embed(self, tokens)
    }

    fn transform(
        &self,
        hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        standard_transform(self, hidden, ctx, cache)
    }
}

// Extension trait - accessors and associated types for typed pipelines
impl TransformerModelExt for ModelWeights {
    type Layer = StandardTransformerBlock;
    type Norm = RmsNorm;

    fn tok_embeddings(&self) -> &Embedding {
        &self.tok_embeddings
    }

    fn layers(&self) -> &[Self::Layer] {
        &self.layers
    }

    fn output_norm(&self) -> &Self::Norm {
        &self.norm
    }

    fn mapper(&self) -> Option<&dyn DeviceMapper> {
        self.mapper
            .as_ref()
            .map(|m| m.as_ref() as &dyn DeviceMapper)
    }

    fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// Object-safe base trait - required lm_head
impl LanguageModel<[KvCache]> for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        standard_lm_head(self, hidden)
    }
}

// Extension trait - output accessor for typed pipelines
impl LanguageModelExt for ModelWeights {
    fn output(&self) -> &Arc<dyn QuantMethod> {
        &self.output
    }
}
