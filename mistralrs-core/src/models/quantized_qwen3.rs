#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3 model using the generic transformer infrastructure.
//!
//! Uses `StandardTransformerBlock` (RmsNorm + CausalAttention + Mlp) composition
//! with Q/K normalization (always present in Qwen3).
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
    TransformContext, TransformerModel, TransformerModelExt,
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

/// Configuration for Qwen3 model loaded from safetensors.
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
    fn hidden_act(&self) -> Activation {
        self.hidden_act
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn quantization_config(&self) -> Option<&QuantizedConfig> {
        self.quantization_config.as_ref()
    }
}

/// Qwen3 model weights using the generic transformer builder infrastructure.
///
/// The model uses pre-norm architecture with:
/// - RMS normalization for attention and FFN
/// - Q/K normalization (Qwen3-specific)
/// - Gated MLP with SiLU activation
/// - RoPE positional embeddings
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
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

        // Load transformer with Q/K normalization customization
        let loaded = load_transformer_from_safetensors(
            cfg,
            TransformerConfig::from_config(cfg),
            vb,
            device,
            &*mapper,
            attention_mechanism,
            dtype,
            layer_range,
            adapter_registry,
            |ctx, builder, weights| {
                // Qwen3-specific: Q/K normalization (always present)
                let q_norm = weights.load_rms_norm(
                    &naming.attn_q_norm(ctx.layer_idx),
                    ctx.rms_norm_eps,
                    ctx.device,
                )?;
                let k_norm = weights.load_rms_norm(
                    &naming.attn_k_norm(ctx.layer_idx),
                    ctx.rms_norm_eps,
                    ctx.device,
                )?;
                Ok(builder.with_qk_norm(Arc::new(RmsNormQkNorm::new(q_norm, k_norm))))
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
        // Verify architecture
        let meta = ct.get_metadata();
        let arch: String = {
            use crate::utils::gguf_metadata::TryValueInto;
            meta.get("general.architecture")
                .cloned()
                .try_value_into()?
        };
        if arch != "qwen3" {
            candle_core::bail!("Expected `qwen3` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata using generic infrastructure
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

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
            weights.load_linear(&naming.output(), config.hidden_size, config.vocab_size, device)?
        } else {
            weights.load_linear(&naming.token_embd(), config.hidden_size, config.vocab_size, device)?
        };

        // Load transformer layers using generic infrastructure with Qwen3-specific customizer
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
                // Qwen3-specific: Add Q/K normalization
                let q_norm = weights.load_rms_norm(
                    &naming.attn_q_norm(ctx.layer_idx),
                    ctx.rms_norm_eps,
                    ctx.device,
                )?;
                let k_norm = weights.load_rms_norm(
                    &naming.attn_k_norm(ctx.layer_idx),
                    ctx.rms_norm_eps,
                    ctx.device,
                )?;
                let qk_norm: Arc<dyn crate::attention::QkNorm> =
                    Arc::new(RmsNormQkNorm::new(q_norm, k_norm));
                Ok(builder.with_qk_norm(qk_norm))
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
        })
    }
}

impl ModelWeights {
    /// Number of transformer layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

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
impl TransformerModel for ModelWeights {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        standard_embed(self, tokens)
    }

    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
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
        self.mapper.as_ref().map(|m| m.as_ref() as &dyn DeviceMapper)
    }

    fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// Object-safe base trait - required lm_head
impl LanguageModel for ModelWeights {
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
