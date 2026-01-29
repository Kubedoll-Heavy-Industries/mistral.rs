#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! SmolLM3 model using the generic transformer infrastructure.
//!
//! Uses `StandardTransformerBlock` (RmsNorm + CausalAttention + Mlp) composition
//! with a special feature: some layers skip RoPE based on `no_rope_layers` config.
//!
//! Supports loading from safetensors format via `FromSafetensors` trait.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};
use serde::{Deserialize, Serialize};

use crate::attention::{IdentityPositionEncoding, PositionEncoding};
use crate::device_map::DeviceMapper;
use crate::layers::{embedding, Activation, RmsNorm, SmolLm3RopeConfig, SmolLm3RotaryEmbedding};
use crate::models::{
    standard_embed, standard_lm_head, standard_transform, LanguageModel, LanguageModelConfig,
    LanguageModelExt, Model, TransformContext, TokenizerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention};
use crate::pipeline::loaders::{
    LayerConfig, StandardTransformerBlock, TransformerConfig, TransformerLayerBuilder,
};
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, true);
serde_default_fn!(usize, default_no_rope_interval, 4);

/// Configuration for SmolLM3 model loaded from safetensors.
/// Mirrors the config.json structure from HuggingFace.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<SmolLm3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    /// Explicit list of layers that skip RoPE (when present).
    pub no_rope_layers: Option<Vec<usize>>,
    /// Interval for layers that skip RoPE (default: every 4th layer uses RoPE).
    /// Applied when `no_rope_layers` is None.
    #[serde(default = "default_no_rope_interval")]
    pub no_rope_layer_interval: usize,
}

impl Config {
    /// Returns a boolean for each layer: true = use RoPE, false = skip RoPE.
    pub fn use_rope_layers(&self) -> Vec<bool> {
        self.no_rope_layers
            .as_ref()
            .map(|explicit| {
                (0..self.num_hidden_layers)
                    .map(|i| !explicit.contains(&i))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                (0..self.num_hidden_layers)
                    .map(|i| (i + 1) % self.no_rope_layer_interval != 0)
                    .collect()
            })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl LanguageModelConfig for Config {
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
        self.rope_theta
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

/// SmolLM3 model weights using the generic transformer builder infrastructure.
///
/// The model uses pre-norm architecture with:
/// - RMS normalization for attention and FFN
/// - Gated MLP with SiLU activation
/// - RoPE positional embeddings (with per-layer control via no_rope_layers)
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    #[allow(dead_code)]
    cfg: ModelConfigMetadata,
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
        // Log quantization info if present
        if let Some(quant_cfg) = cfg.quantization_config.as_ref() {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }

        let vb_m = vb.pp("model");
        let use_rope_layers = cfg.use_rope_layers();

        // Load embedding weights
        let tok_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &cfg.quantization_config,
        )?;

        // Load output norm
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Load output weights (may be tied to embeddings)
        let output: Arc<dyn QuantMethod> = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                vb.pp("lm_head"),
            )?
        } else {
            mistralrs_quant::ReplicatedLayer::from_linear(candle_nn::Linear::new(
                tok_embeddings.embeddings().clone(),
                None,
            ))?
        };

        // Determine layer range
        let config = TransformerConfig::from_config(cfg);
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(config.num_layers))
            .unwrap_or(config.num_layers);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < config.num_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                config.num_layers
            );
        }

        // Create RoPE embeddings for each device location
        // SmolLM3 uses SmolLm3RotaryEmbedding which wraps standard RoPE with config options
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<SmolLm3RotaryEmbedding>> =
            HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(SmolLm3RotaryEmbedding::new_llama3(
                    dtype,
                    cfg,
                    layer_device,
                    true, // is_gptx
                )?));
            }
        }

        // Identity position encoding for layers that skip RoPE
        let identity_rope: Arc<dyn PositionEncoding> = Arc::new(IdentityPositionEncoding);

        // Extract dimensions from config for loading
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let quant_cfg = cfg.quantization_config.clone();

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_loaded_layers);
        let vb_l = vb_m.pp("layers");

        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);

            // Get position encoding for this layer (RoPE or identity)
            let rope: Arc<dyn PositionEncoding> = if use_rope_layers[layer_idx] {
                ropes
                    .get(&layer_device.location())
                    .expect("No RoPE for device location!")
                    .clone() as Arc<dyn PositionEncoding>
            } else {
                identity_rope.clone()
            };

            let vb_layer = vb_l.pp(layer_idx);
            let vb_attn = vb_layer.pp("self_attn");
            let vb_mlp = vb_layer.pp("mlp");

            // Load layer weights
            let q_proj = mistralrs_quant::linear_no_bias(
                hidden_size,
                num_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("q_proj"),
            )?;
            let k_proj = mistralrs_quant::linear_no_bias(
                hidden_size,
                num_kv_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("k_proj"),
            )?;
            let v_proj = mistralrs_quant::linear_no_bias(
                hidden_size,
                num_kv_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("v_proj"),
            )?;
            let o_proj = mistralrs_quant::linear_no_bias(
                num_heads * head_dim,
                hidden_size,
                &quant_cfg,
                vb_attn.pp("o_proj"),
            )?;

            // Load MLP weights
            let gate_proj = mistralrs_quant::linear_no_bias(
                hidden_size,
                intermediate_size,
                &quant_cfg,
                vb_mlp.pp("gate_proj"),
            )?;
            let up_proj = mistralrs_quant::linear_no_bias(
                hidden_size,
                intermediate_size,
                &quant_cfg,
                vb_mlp.pp("up_proj"),
            )?;
            let down_proj = mistralrs_quant::linear_no_bias(
                intermediate_size,
                hidden_size,
                &quant_cfg,
                vb_mlp.pp("down_proj"),
            )?;

            // Load normalization layers
            let attn_norm =
                RmsNorm::new(hidden_size, config.rms_norm_eps, vb_layer.pp("input_layernorm"))?;
            let ffn_norm = RmsNorm::new(
                hidden_size,
                config.rms_norm_eps,
                vb_layer.pp("post_attention_layernorm"),
            )?;

            // Build layer config
            let layer_config =
                LayerConfig::new(num_heads, num_kv_heads, head_dim, config.hidden_act);

            // Create builder with loaded weights
            let mut builder = TransformerLayerBuilder::new(layer_config)
                .q_proj(q_proj)
                .k_proj(k_proj)
                .v_proj(v_proj)
                .o_proj(o_proj)
                .gate_proj(gate_proj)
                .up_proj(up_proj)
                .down_proj(down_proj)
                .attn_norm(attn_norm)
                .ffn_norm(ffn_norm)
                .rope(rope)
                .with_attn_dtype(dtype);

            // Add adapter registry for per-request LoRA switching
            if let Some(ref registry) = adapter_registry {
                builder = builder.with_adapter_registry(registry.clone(), layer_idx);
            }

            // Add paged attention if enabled
            if let AttentionImplementation::PagedAttention = attention_mechanism {
                builder =
                    builder.with_paged_attn(PagedAttention::new(head_dim, layer_device, None)?);
            }

            // Build the layer
            layers.push(builder.build()?);
        }

        // Build model config metadata for compatibility
        let model_cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_attn_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            sliding_window: None,
            k_head_dim: config.head_dim,
            v_head_dim: config.head_dim,
        };

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            mapper: Some(mapper),
            dtype,
            cfg: model_cfg,
        })
    }
}

impl ModelWeights {
    /// Number of transformer layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
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
        self.cfg.k_head_dim * self.cfg.num_kv_heads
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

// =============================================================================
// Legacy Compatibility Type Alias
// =============================================================================

/// Legacy type alias for compatibility with existing code.
#[deprecated(since = "0.8.0", note = "Use ModelWeights directly")]
pub type SmolLm3 = ModelWeights;
