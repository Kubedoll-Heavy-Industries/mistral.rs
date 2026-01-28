#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Llama model implementation using the generic transformer infrastructure.
//!
//! This module provides a unified `Llama` struct that works with both GGUF and
//! safetensors formats. Position encoding (standard RoPE or Llama3 RoPE with
//! scaling) is handled via type-erased `Arc<dyn PositionEncoding>`.
//!
//! # Type Aliases
//!
//! For backward compatibility, type aliases are provided:
//! - `LlamaModel` - Alias for `Llama` (GGUF loading uses standard RoPE)
//! - `Llama3Model` - Alias for `Llama` (safetensors loading uses Llama3 RoPE)
//!
//! # Supported Variants
//!
//! - LLaMA, LLaMA 2, LLaMA 3, Code Llama (dense models)
//! - For MoE variants (Mixtral), see the `mixtral` module

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};
use serde::{Deserialize, Serialize};

use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{Activation, Llama3RopeConfig, RmsNorm};
use crate::models::{
    standard_embed, standard_lm_head, standard_transform, LanguageModel, LanguageModelConfig,
    LanguageModelExt, Model, TransformContext, TransformerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::loaders::{
    load_transformer_from_safetensors, load_transformer_layers, GgufNaming, GgufWeightSource,
    SafetensorsNaming, StandardTransformerBlock, TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::{KvCache, NormalLoadingMetadata};
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config::{self as ModelConfig, FromSafetensors};
use crate::utils::unvarbuilder::UnVarBuilder;

serde_default_fn!(bool, word_emb_default, false);

// =============================================================================
// Type Aliases (Backward Compatibility)
// =============================================================================

/// Basic Llama model - alias for `Llama`.
///
/// When loading from GGUF, uses standard rotary embeddings.
/// When loading from safetensors, uses Llama3 rotary embeddings with scaling.
pub type LlamaModel = Llama;

/// Llama 3 model - alias for `Llama`.
///
/// Identical to `LlamaModel`. The position encoding variant is determined
/// at load time based on the model format and configuration, not by the type.
pub type Llama3Model = Llama;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for Llama models (from config.json for safetensors).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl Config {
    /// Compute head dimension from hidden size and num attention heads.
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

// =============================================================================
// Model Structure
// =============================================================================

/// Llama language model using the generic transformer infrastructure.
///
/// This model uses `StandardTransformerBlock` internally, with position
/// encoding handled via type-erased `Arc<dyn PositionEncoding>`. The specific
/// RoPE variant (standard or Llama3 with scaling) is determined at load time.
pub struct Llama {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    output_norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    device: Device,
    max_seq_len: usize,
    dtype: DType,
    cfg: ModelConfigMetadata,
}

// =============================================================================
// FromGGUF Implementation
// =============================================================================

impl ModelConfig::FromGGUF for Llama {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
        adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        // Extract configuration from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: "llama",
            metadata: ct.get_metadata(),
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Create weight source and naming
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
        let output_norm =
            weights.load_rms_norm(&naming.output_norm(), config.rms_norm_eps, device)?;

        // Load output weights (tie to embeddings if not present)
        let vocab_size = tok_embeddings.embeddings().dim(0)?;
        let output = if weights.has_tensor(&naming.output()) {
            weights.load_linear(&naming.output(), config.hidden_size, vocab_size, device)?
        } else {
            weights.load_linear(&naming.token_embd(), config.hidden_size, vocab_size, device)?
        };

        // Load transformer layers using generic infrastructure
        // Llama uses standard transformer blocks with no customization
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
            |_ctx, builder, _weights| Ok(builder), // No customization needed for Llama
        )?;

        // Build model config metadata for compatibility
        let cfg = ModelConfigMetadata {
            max_seq_len: config.max_seq_len,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_attn_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            sliding_window: config.sliding_window,
            k_head_dim: config.head_dim,
            v_head_dim: config.head_dim,
        };

        Ok(Self {
            tok_embeddings,
            layers,
            output_norm,
            output,
            mapper: Some(mapper),
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            dtype,
            cfg,
        })
    }
}

// =============================================================================
// FromSafetensors Implementation
// =============================================================================

impl ModelConfig::FromSafetensors for Llama {
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
        let _naming = SafetensorsNaming;

        // Build TransformerConfig from the config
        let config = TransformerConfig::from_config(cfg);

        // Load transformer using generic infrastructure
        let loaded = load_transformer_from_safetensors(
            cfg,
            config.clone(),
            vb,
            device,
            &*mapper,
            attention_mechanism,
            dtype,
            layer_range,
            adapter_registry,
            |_ctx, builder, _weights| Ok(builder), // No customization needed for Llama
        )?;

        // Build model config metadata for compatibility
        let model_cfg = ModelConfigMetadata {
            max_seq_len: config.max_seq_len,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_attn_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            sliding_window: config.sliding_window,
            k_head_dim: config.head_dim,
            v_head_dim: config.head_dim,
        };

        Ok(Self {
            tok_embeddings: loaded.tok_embeddings,
            layers: loaded.layers,
            output_norm: loaded.output_norm,
            output: loaded.output,
            mapper: Some(mapper),
            device: device.clone(),
            max_seq_len: loaded.max_seq_len,
            dtype,
            cfg: model_cfg,
        })
    }
}

// =============================================================================
// Model Trait Implementations
// =============================================================================

impl Model for Llama {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TransformerModel for Llama {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
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

impl TransformerModelExt for Llama {
    type Layer = StandardTransformerBlock;
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
        self.mapper
            .as_ref()
            .map(|m| m.as_ref() as &dyn DeviceMapper)
    }

    fn model_dtype(&self) -> DType {
        self.dtype
    }
}

impl LanguageModel for Llama {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        standard_lm_head(self, hidden)
    }
}

impl LanguageModelExt for Llama {
    fn output(&self) -> &Arc<dyn QuantMethod> {
        &self.output
    }
}

// =============================================================================
// Deprecated Methods (Vision Model Compatibility)
// =============================================================================
//
// These methods exist for backward compatibility with vision models that
// compose with Llama (e.g., Idefics3, LLaVA). Vision models should migrate
// to use the standard TransformerModel/LanguageModel trait interface instead.

impl Llama {
    /// Constructor for vision models using the legacy NormalLoadingMetadata pattern.
    ///
    /// **Deprecated**: Vision models should migrate to use `FromSafetensors` trait
    /// and the standard TransformerModel interface.
    #[deprecated(
        since = "0.8.0",
        note = "Vision models should migrate to use the FromSafetensors trait and TransformerModel interface."
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        _vb_lm_head: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Build VarBuilder for full model
        let vb = vb_m.clone();
        let mapper = normal_loading_metadata.mapper;
        let device = &normal_loading_metadata.real_device;
        let dtype = vb.dtype();

        // Use FromSafetensors implementation
        Self::from_safetensors(
            cfg,
            vb,
            device,
            mapper,
            attention_mechanism,
            dtype,
            None,
            None,
        )
    }

    /// Get input embeddings for tokens.
    ///
    /// **Deprecated**: Use `TransformerModel::embed()` instead.
    #[deprecated(
        since = "0.8.0",
        note = "Use TransformerModel::embed() instead. Vision models should migrate to the standard trait interface."
    )]
    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed(input_ids)
    }

    /// Forward pass starting from pre-computed embeddings.
    ///
    /// **Deprecated**: Vision models should use `TransformerModel::transform()` and
    /// `LanguageModel::lm_head()` separately, managing their own cache.
    #[deprecated(
        since = "0.8.0",
        note = "Use TransformerModel::transform() + LanguageModel::lm_head() instead. Vision models should migrate to the standard trait interface."
    )]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn forward_embeds(
        &self,
        _input_ids: &Tensor,
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        metadata: Option<(
            Vec<(Tensor, Tensor)>,
            &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
        )>,
        _flash_params: &crate::pipeline::text_models_inputs_processor::FlashParams,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let seq_len = input_embeds.dim(1)?;
        let position_offset = seqlen_offsets.first().copied().unwrap_or(0);

        let paged_attn_ctx =
            metadata.map(|(kv_cache, meta)| crate::models::PagedAttentionContext {
                kv_cache,
                metadata: meta,
            });

        let ctx = TransformContext {
            seq_len,
            position_offset,
            paged_attn: paged_attn_ctx.as_ref(),
            flash_params: None,
            position_ids: None,
        };

        let hidden = self.transform(input_embeds, &ctx, cache)?;
        self.lm_head(hidden)
    }

    /// Get quantizable layers for ISQ.
    ///
    /// **Deprecated**: Vision models should implement IsqModel directly.
    #[deprecated(
        since = "0.8.0",
        note = "Vision models should implement IsqModel directly using the TransformerModelExt interface."
    )]
    #[allow(clippy::type_complexity)]
    pub fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mapper = self
            .mapper
            .as_ref()
            .expect("Model must have a mapper for get_layers");

        let mut layers = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Each StandardTransformerBlock has attention (q, k, v, o) and MLP (gate, up, down)
            let (q, k, v, o) = layer.attention.projections_mut();
            layers.push((q, Some(i)));
            layers.push((k, Some(i)));
            layers.push((v, Some(i)));
            layers.push((o, Some(i)));
            layers.push((&mut layer.ffn.gate, Some(i)));
            layers.push((&mut layer.ffn.up, Some(i)));
            layers.push((&mut layer.ffn.down, Some(i)));
        }
        layers.push((&mut self.output, None));

        (layers, &**mapper)
    }

    /// Get residual tensors for UQFF serialization.
    ///
    /// **Deprecated**: Vision models should implement residual_tensors directly.
    #[deprecated(
        since = "0.8.0",
        note = "Vision models should implement residual_tensors directly using the TransformerModelExt interface."
    )]
    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.tok_embeddings);
        uvb_m.pp("norm").add(&self.output_norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.attn_norm);
            uvb_l.pp("post_attention_layernorm").add(&layer.ffn_norm);
        }

        uvb_m.to_safetensors()
    }

    /// Get model config metadata.
    ///
    /// **Deprecated**: Use TransformerModel methods directly.
    #[deprecated(
        since = "0.8.0",
        note = "Use TransformerModel::num_layers(), max_seq_len(), etc. directly."
    )]
    pub fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}
