//! Generic transformer loading infrastructure.
//!
//! This module provides abstractions for loading transformer models from
//! any weight format (GGUF, safetensors, etc.) with model-specific customization.
//!
//! # Architecture
//!
//! ```text
//! Config Source (GGUF metadata / JSON)
//!     ↓
//! TransformerConfig (common struct)
//!     ↓
//! load_transformer_layers() + WeightSource + customizer
//!     ↓
//! Vec<TransformerBlock>
//! ```
//!
//! # Example
//!
//! ```ignore
//! // Parse config from GGUF metadata
//! let config = TransformerConfig::from_gguf_metadata(metadata)?;
//!
//! // Load layers with model-specific customization
//! let layers = load_transformer_layers(
//!     &config,
//!     &mut gguf_weights,
//!     layer_range,
//!     &mapper,
//!     attention_mechanism,
//!     dtype,
//!     |layer_idx, builder, weights| {
//!         // Qwen3-specific: add Q/K norm
//!         let q_norm = weights.load_norm(&format!("blk.{layer_idx}.attn_q_norm"), config.rms_norm_eps)?;
//!         let k_norm = weights.load_norm(&format!("blk.{layer_idx}.attn_k_norm"), config.rms_norm_eps)?;
//!         Ok(builder.with_qk_norm(Arc::new(RmsNormQkNorm::new(q_norm, k_norm))))
//!     },
//! )?;
//! ```

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use candle_core::{DType, Device, Result};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::QuantMethod;

use crate::attention::{AttentionConfig, CausalAttention, PositionEncoding, QkNorm};
use crate::device_map::DeviceMapper;
use crate::layers::{Activation, Mlp, RmsNorm, RotaryEmbedding, TransformerBlock};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// ============================================================================
// Common Transformer Configuration
// ============================================================================

/// Common transformer configuration extracted from any source.
///
/// This struct captures the architectural hyperparameters needed to load
/// any decoder-only transformer model. It can be populated from:
/// - GGUF metadata via `from_gguf_metadata()`
/// - JSON config via `From<T>` implementations
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    // Core dimensions
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    // Sequence/position
    pub max_seq_len: usize,
    pub rope_theta: f32,

    // Normalization
    pub rms_norm_eps: f64,

    // Activation
    pub hidden_act: Activation,

    // Optional features
    pub tie_word_embeddings: bool,
    pub sliding_window: Option<usize>,
    /// Whether to load optional biases for Q/K/V attention projections (Qwen2-style).
    /// When true, `load_linear_with_optional_bias` is used for Q/K/V projections.
    pub use_attention_bias: bool,
}

/// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

impl TransformerConfig {
    /// Create config with required fields, optional fields defaulted.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        rms_norm_eps: f64,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim: hidden_size / num_heads,
            num_layers,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta: 10_000.0,
            rms_norm_eps,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            sliding_window: None,
            use_attention_bias: false,
        }
    }

    /// Parse transformer config from GGUF metadata.
    ///
    /// This extracts the common architectural hyperparameters that most
    /// decoder-only transformer models share. Model-specific fields
    /// (like Qwen3's Q/K norm) are handled via the customizer callback.
    ///
    /// # Arguments
    /// * `metadata` - GGUF ContentMetadata with architecture prefix set
    ///
    /// # Standard GGUF Fields
    /// Required:
    /// - `{arch}.embedding_length` -> hidden_size
    /// - `{arch}.attention.head_count` -> num_heads
    /// - `{arch}.attention.head_count_kv` -> num_kv_heads
    /// - `{arch}.block_count` -> num_layers
    /// - `{arch}.attention.layer_norm_rms_epsilon` -> rms_norm_eps
    ///
    /// Optional:
    /// - `{arch}.context_length` -> max_seq_len (default: 4096)
    /// - `{arch}.rope.freq_base` -> rope_theta (default: 10000.0)
    /// - `{arch}.attention.key_length` -> head_dim (default: hidden_size / num_heads)
    /// - `{arch}.feed_forward_length` -> intermediate_size
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_gguf_metadata(
        metadata: &crate::utils::gguf_metadata::ContentMetadata,
    ) -> std::result::Result<Self, anyhow::Error> {
        // Validate required fields exist
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        metadata.has_required_keys(&required)?;

        // Extract required fields
        let hidden_size = metadata.get_value::<u32>("embedding_length")? as usize;
        let num_heads = metadata.get_value::<u32>("attention.head_count")? as usize;
        let num_kv_heads = metadata.get_value::<u32>("attention.head_count_kv")? as usize;
        let num_layers = metadata.get_value::<u32>("block_count")? as usize;
        let rms_norm_eps = metadata.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64;

        // Extract optional fields with defaults
        let max_seq_len = metadata
            .get_value::<u64>("context_length")
            .ok()
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);

        let rope_theta = metadata
            .get_value::<f32>("rope.freq_base")
            .ok()
            .unwrap_or(10_000.0);

        let head_dim = metadata
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(hidden_size / num_heads);

        // Verify key_length == value_length (standard for most models)
        let value_length = metadata
            .get_value::<u32>("attention.value_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(head_dim);

        if head_dim != value_length {
            anyhow::bail!(
                "Expected key_length == value_length, got {head_dim} != {value_length}"
            );
        }

        // intermediate_size often not in GGUF metadata, use 0 as sentinel
        let intermediate_size = metadata
            .get_value::<u32>("feed_forward_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);

        // vocab_size from embedding tensor (caller may need to set this separately)
        let vocab_size = 0; // Will be set from embedding tensor dimensions

        Ok(Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            hidden_act: Activation::Silu, // Most models use SiLU/Swish
            tie_word_embeddings: false,
            sliding_window: None,
            use_attention_bias: false,
        })
    }

    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    pub fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta = theta;
        self
    }

    pub fn with_hidden_act(mut self, act: Activation) -> Self {
        self.hidden_act = act;
        self
    }

    pub fn with_tie_word_embeddings(mut self, tie: bool) -> Self {
        self.tie_word_embeddings = tie;
        self
    }

    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Enable optional attention biases for Q/K/V projections (Qwen2-style).
    pub fn with_attention_bias(mut self) -> Self {
        self.use_attention_bias = true;
        self
    }
}

// ============================================================================
// Weight Source Abstraction
// ============================================================================

/// Abstraction over weight sources (GGUF, safetensors, etc.).
///
/// This trait provides a uniform interface for loading weights regardless
/// of the underlying format. Implementations handle format-specific details
/// like tensor naming conventions and quantization.
pub trait WeightSource {
    /// Load a linear layer weight as quantized method.
    fn load_linear(&mut self, name: &str, device: &Device) -> Result<Arc<dyn QuantMethod>>;

    /// Load a linear layer with an optional bias tensor.
    ///
    /// Bias tensor name is derived by replacing `.weight` suffix with `.bias`.
    /// If the bias tensor exists, it will be dequantized and included in the result.
    fn load_linear_with_optional_bias(
        &mut self,
        weight_name: &str,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>>;

    /// Load an embedding layer.
    fn load_embedding(
        &mut self,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> Result<Embedding>;

    /// Load an RMS normalization layer.
    fn load_rms_norm(&mut self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm>;

    /// Load a LayerNorm layer (weight and bias).
    ///
    /// Loads `{base_name}.weight` and `{base_name}.bias`, dequantizes both,
    /// and creates a `candle_nn::LayerNorm` with the given epsilon.
    fn load_layer_norm(&mut self, base_name: &str, eps: f64, device: &Device) -> Result<LayerNorm>;

    /// Check if a tensor exists.
    fn has_tensor(&self, name: &str) -> bool;
}

// ============================================================================
// GGUF Weight Source Implementation
// ============================================================================

use crate::gguf::Content;
use mistralrs_quant::{GgufMatMul, QuantMethodConfig};

/// GGUF weight source - wraps Content<R> for loading from GGUF files.
///
/// Uses two lifetimes:
/// - `'a` - lifetime of the mutable reference to Content
/// - `'c` - lifetime of Content's internal data (file reference, etc.)
pub struct GgufWeightSource<'a, 'c, R: std::io::Seek + std::io::Read> {
    content: &'a mut Content<'c, R>,
}

impl<'a, 'c, R: std::io::Seek + std::io::Read> GgufWeightSource<'a, 'c, R> {
    pub fn new(content: &'a mut Content<'c, R>) -> Self {
        Self { content }
    }
}

impl<R: std::io::Seek + std::io::Read> WeightSource for GgufWeightSource<'_, '_, R> {
    fn load_linear(&mut self, name: &str, device: &Device) -> Result<Arc<dyn QuantMethod>> {
        let qtensor = self.content.tensor(name, device)?;
        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(qtensor),
            b: None,
        })?))
    }

    fn load_linear_with_optional_bias(
        &mut self,
        weight_name: &str,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let qtensor = self.content.tensor(weight_name, device)?;

        // Derive bias name by replacing `.weight` suffix with `.bias`
        let bias_name = weight_name.replace(".weight", ".bias");
        let bias = if self.content.has_tensor(&bias_name) {
            let bias_qtensor = self.content.tensor(&bias_name, device)?;
            Some(bias_qtensor.dequantize(device)?)
        } else {
            None
        };

        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(qtensor),
            b: bias,
        })?))
    }

    fn load_embedding(
        &mut self,
        name: &str,
        _vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> Result<Embedding> {
        let qtensor = self.content.tensor(name, device)?;
        let tensor = qtensor.dequantize(device)?;
        Ok(Embedding::new(tensor, hidden_size))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn load_rms_norm(&mut self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm> {
        let qtensor = self.content.tensor(name, device)?;
        RmsNorm::from_qtensor(qtensor, eps as f32)
    }

    fn load_layer_norm(&mut self, base_name: &str, eps: f64, device: &Device) -> Result<LayerNorm> {
        let weight_name = format!("{base_name}.weight");
        let bias_name = format!("{base_name}.bias");

        let weight_qtensor = self.content.tensor(&weight_name, device)?;
        let bias_qtensor = self.content.tensor(&bias_name, device)?;

        let weight = weight_qtensor.dequantize(device)?;
        let bias = bias_qtensor.dequantize(device)?;

        Ok(LayerNorm::new(weight, bias, eps))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.content.has_tensor(name)
    }
}

/// Configuration for a transformer layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    /// Sliding window size (None for full attention)
    pub sliding_window: Option<usize>,
    /// Softcap for attention logits (Gemma2-specific)
    pub softcap: Option<f32>,
}

impl LayerConfig {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, hidden_act: Activation) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_act,
            sliding_window: None,
            softcap: None,
        }
    }

    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    pub fn with_softcap(mut self, cap: f32) -> Self {
        self.softcap = Some(cap);
        self
    }
}

/// Builder for assembling transformer layers from loaded weights.
///
/// This separates weight loading (format-specific) from layer assembly (generic).
/// Models load weights in their format-specific way, then use this builder to
/// assemble the standard transformer layer structure.
pub struct TransformerLayerBuilder {
    config: LayerConfig,
    // Attention weights
    q_proj: Option<Arc<dyn QuantMethod>>,
    k_proj: Option<Arc<dyn QuantMethod>>,
    v_proj: Option<Arc<dyn QuantMethod>>,
    o_proj: Option<Arc<dyn QuantMethod>>,
    // MLP weights
    gate_proj: Option<Arc<dyn QuantMethod>>,
    up_proj: Option<Arc<dyn QuantMethod>>,
    down_proj: Option<Arc<dyn QuantMethod>>,
    // Norms
    attn_norm: Option<RmsNorm>,
    ffn_norm: Option<RmsNorm>,
    // Position encoding
    rope: Option<Arc<dyn PositionEncoding>>,
    // Optional features
    qk_norm: Option<Arc<dyn QkNorm>>,
    paged_attn: Option<PagedAttention>,
    attn_dtype: Option<DType>,
}

impl TransformerLayerBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: LayerConfig) -> Self {
        Self {
            config,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            attn_norm: None,
            ffn_norm: None,
            rope: None,
            qk_norm: None,
            paged_attn: None,
            attn_dtype: None,
        }
    }

    // ========================================================================
    // Weight setters - take Arc<dyn QuantMethod> from any source
    // ========================================================================

    pub fn q_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.q_proj = Some(w);
        self
    }

    pub fn k_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.k_proj = Some(w);
        self
    }

    pub fn v_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.v_proj = Some(w);
        self
    }

    pub fn o_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.o_proj = Some(w);
        self
    }

    pub fn gate_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.gate_proj = Some(w);
        self
    }

    pub fn up_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.up_proj = Some(w);
        self
    }

    pub fn down_proj(mut self, w: Arc<dyn QuantMethod>) -> Self {
        self.down_proj = Some(w);
        self
    }

    pub fn attn_norm(mut self, norm: RmsNorm) -> Self {
        self.attn_norm = Some(norm);
        self
    }

    pub fn ffn_norm(mut self, norm: RmsNorm) -> Self {
        self.ffn_norm = Some(norm);
        self
    }

    pub fn rope(mut self, rope: Arc<dyn PositionEncoding>) -> Self {
        self.rope = Some(rope);
        self
    }

    // ========================================================================
    // Optional features
    // ========================================================================

    /// Add Q/K normalization (Qwen3-specific).
    pub fn with_qk_norm(mut self, qk_norm: Arc<dyn QkNorm>) -> Self {
        self.qk_norm = Some(qk_norm);
        self
    }

    /// Add paged attention support.
    pub fn with_paged_attn(mut self, paged_attn: PagedAttention) -> Self {
        self.paged_attn = Some(paged_attn);
        self
    }

    /// Set target dtype for attention computation (for quantized models).
    pub fn with_attn_dtype(mut self, dtype: DType) -> Self {
        self.attn_dtype = Some(dtype);
        self
    }

    // ========================================================================
    // Build
    // ========================================================================

    /// Build the transformer layer from the provided weights.
    ///
    /// Returns an error if any required weights are missing.
    pub fn build(self) -> Result<TransformerBlock<RmsNorm, CausalAttention, Mlp>> {
        let q_proj = self.q_proj.ok_or_else(|| candle_core::Error::Msg("q_proj not set".into()))?;
        let k_proj = self.k_proj.ok_or_else(|| candle_core::Error::Msg("k_proj not set".into()))?;
        let v_proj = self.v_proj.ok_or_else(|| candle_core::Error::Msg("v_proj not set".into()))?;
        let o_proj = self.o_proj.ok_or_else(|| candle_core::Error::Msg("o_proj not set".into()))?;
        let gate_proj = self.gate_proj.ok_or_else(|| candle_core::Error::Msg("gate_proj not set".into()))?;
        let up_proj = self.up_proj.ok_or_else(|| candle_core::Error::Msg("up_proj not set".into()))?;
        let down_proj = self.down_proj.ok_or_else(|| candle_core::Error::Msg("down_proj not set".into()))?;
        let attn_norm = self.attn_norm.ok_or_else(|| candle_core::Error::Msg("attn_norm not set".into()))?;
        let ffn_norm = self.ffn_norm.ok_or_else(|| candle_core::Error::Msg("ffn_norm not set".into()))?;
        let rope = self.rope.ok_or_else(|| candle_core::Error::Msg("rope not set".into()))?;

        // Build attention config
        let mut attn_config = AttentionConfig::new(
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        );
        if let Some(window) = self.config.sliding_window {
            attn_config = attn_config.with_sliding_window(window);
        }
        if let Some(cap) = self.config.softcap {
            attn_config = attn_config.with_softcap(cap);
        }

        // Build attention
        let mut attention = CausalAttention::new(attn_config, q_proj, k_proj, v_proj, o_proj, rope);
        if let Some(qk_norm) = self.qk_norm {
            attention = attention.with_qk_norm(qk_norm);
        }
        if let Some(paged_attn) = self.paged_attn {
            attention = attention.with_paged_attn(paged_attn);
        }
        if let Some(dtype) = self.attn_dtype {
            attention = attention.with_attn_dtype(dtype);
        }

        // Build MLP
        let mlp = Mlp::from_weights(gate_proj, up_proj, down_proj, self.config.hidden_act);

        // Assemble layer
        Ok(TransformerBlock::new(attn_norm, attention, ffn_norm, mlp))
    }
}

// ============================================================================
// Generic Layer Loading
// ============================================================================

/// Standard transformer block type alias.
pub type StandardTransformerBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;

/// Context passed to the layer customizer closure.
///
/// This provides access to per-layer information needed for
/// model-specific customization (like loading Q/K norm tensors).
pub struct LayerCustomizerContext<'a> {
    /// Layer index (global, not relative to loaded range)
    pub layer_idx: usize,
    /// Device for this layer (after device mapping)
    pub device: &'a Device,
    /// RMS norm epsilon from config
    pub rms_norm_eps: f64,
}

/// Tensor naming convention for different formats.
pub trait TensorNaming {
    /// Get the tensor name for a layer's attention Q projection.
    fn attn_q(&self, layer_idx: usize) -> String;
    fn attn_k(&self, layer_idx: usize) -> String;
    fn attn_v(&self, layer_idx: usize) -> String;
    fn attn_output(&self, layer_idx: usize) -> String;
    fn ffn_gate(&self, layer_idx: usize) -> String;
    fn ffn_up(&self, layer_idx: usize) -> String;
    fn ffn_down(&self, layer_idx: usize) -> String;
    fn attn_norm(&self, layer_idx: usize) -> String;
    fn ffn_norm(&self, layer_idx: usize) -> String;
    fn token_embd(&self) -> String;
    fn output_norm(&self) -> String;
    fn output(&self) -> String;
}

/// GGUF tensor naming convention.
pub struct GgufNaming;

impl TensorNaming for GgufNaming {
    fn attn_q(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_q.weight")
    }
    fn attn_k(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_k.weight")
    }
    fn attn_v(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_v.weight")
    }
    fn attn_output(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_output.weight")
    }
    fn ffn_gate(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_gate.weight")
    }
    fn ffn_up(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_up.weight")
    }
    fn ffn_down(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_down.weight")
    }
    fn attn_norm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_norm.weight")
    }
    fn ffn_norm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_norm.weight")
    }
    fn token_embd(&self) -> String {
        "token_embd.weight".to_string()
    }
    fn output_norm(&self) -> String {
        "output_norm.weight".to_string()
    }
    fn output(&self) -> String {
        "output.weight".to_string()
    }
}

impl GgufNaming {
    /// Qwen3-specific: Q norm tensor name
    pub fn attn_q_norm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_q_norm.weight")
    }

    /// Qwen3-specific: K norm tensor name
    pub fn attn_k_norm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_k_norm.weight")
    }

    /// Starcoder2-specific: attention Q bias tensor name
    pub fn attn_q_bias(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_q.bias")
    }

    /// Starcoder2-specific: attention K bias tensor name
    pub fn attn_k_bias(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_k.bias")
    }

    /// Starcoder2-specific: attention V bias tensor name
    pub fn attn_v_bias(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_v.bias")
    }

    /// Starcoder2-specific: attention output bias tensor name
    pub fn attn_output_bias(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_output.bias")
    }
}

/// Safetensors (HuggingFace) tensor naming convention.
pub struct SafetensorsNaming;

impl TensorNaming for SafetensorsNaming {
    fn attn_q(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.self_attn.q_proj")
    }
    fn attn_k(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.self_attn.k_proj")
    }
    fn attn_v(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.self_attn.v_proj")
    }
    fn attn_output(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.self_attn.o_proj")
    }
    fn ffn_gate(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.mlp.gate_proj")
    }
    fn ffn_up(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.mlp.up_proj")
    }
    fn ffn_down(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.mlp.down_proj")
    }
    fn attn_norm(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.input_layernorm")
    }
    fn ffn_norm(&self, layer_idx: usize) -> String {
        format!("model.layers.{layer_idx}.post_attention_layernorm")
    }
    fn token_embd(&self) -> String {
        "model.embed_tokens".to_string()
    }
    fn output_norm(&self) -> String {
        "model.norm".to_string()
    }
    fn output(&self) -> String {
        "lm_head".to_string()
    }
}

/// Load transformer layers with model-specific customization.
///
/// This is the main entry point for generic transformer loading. It:
/// 1. Handles layer range for pipeline parallelism
/// 2. Creates RoPE embeddings per device
/// 3. Loads each layer's weights via the WeightSource
/// 4. Calls the customizer for model-specific features (Q/K norm, etc.)
/// 5. Builds each layer using TransformerLayerBuilder
///
/// # Arguments
/// * `config` - Common transformer configuration
/// * `weights` - Weight source (GGUF, safetensors, etc.)
/// * `naming` - Tensor naming convention
/// * `layer_range` - Optional range of layers to load (for pipeline parallelism)
/// * `mapper` - Device mapper for multi-GPU
/// * `device` - Default device
/// * `attention_mechanism` - Eager or paged attention
/// * `dtype` - Target dtype for attention
/// * `customizer` - Closure to add model-specific features to each layer.
///   Receives `(context, builder, weights)` where context provides layer-specific info.
///
/// # Returns
/// Vector of assembled transformer blocks.
///
/// # Example
///
/// ```ignore
/// // Load Qwen3 with Q/K norm customization
/// let layers = load_transformer_layers(
///     &config,
///     &mut weights,
///     &GgufNaming,
///     layer_range,
///     &*mapper,
///     device,
///     attention_mechanism,
///     dtype,
///     |ctx, builder, weights| {
///         // Load Qwen3-specific Q/K norm
///         let q_norm = weights.load_rms_norm(
///             &GgufNaming.attn_q_norm(ctx.layer_idx),
///             ctx.rms_norm_eps,
///             ctx.device,
///         )?;
///         let k_norm = weights.load_rms_norm(
///             &GgufNaming.attn_k_norm(ctx.layer_idx),
///             ctx.rms_norm_eps,
///             ctx.device,
///         )?;
///         let qk_norm = Arc::new(RmsNormQkNorm::new(q_norm, k_norm));
///         Ok(builder.with_qk_norm(qk_norm))
///     },
/// )?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn load_transformer_layers<W, N, F>(
    config: &TransformerConfig,
    weights: &mut W,
    naming: &N,
    layer_range: Option<Range<usize>>,
    mapper: &dyn DeviceMapper,
    device: &Device,
    attention_mechanism: AttentionImplementation,
    dtype: DType,
    mut customizer: F,
) -> Result<Vec<StandardTransformerBlock>>
where
    W: WeightSource,
    N: TensorNaming,
    F: FnMut(LayerCustomizerContext<'_>, TransformerLayerBuilder, &mut W) -> Result<TransformerLayerBuilder>,
{
    // Determine layer range
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
    let mut ropes: HashMap<candle_core::DeviceLocation, Arc<RotaryEmbedding>> = HashMap::new();
    for layer_idx in layer_start..layer_end {
        let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
        if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(layer_device.location()) {
            e.insert(Arc::new(RotaryEmbedding::new(
                config.rope_theta,
                config.head_dim,
                config.max_seq_len,
                layer_device,
                true, // is_gptx
                DType::F32,
            )?));
        }
    }

    // Load layers
    let mut layers = Vec::with_capacity(num_loaded_layers);

    for layer_idx in NiceProgressBar::<_, 'b'>(
        layer_start..layer_end,
        "Loading repeating layers",
        &new_multi_progress(),
    ) {
        let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);

        // Get RoPE for this device
        let rotary = ropes
            .get(&layer_device.location())
            .expect("No RoPE for device location!")
            .clone();

        // Load layer weights
        // Use optional bias loading for Q/K/V if enabled (Qwen2-style)
        let (q_proj, k_proj, v_proj) = if config.use_attention_bias {
            (
                weights.load_linear_with_optional_bias(&naming.attn_q(layer_idx), layer_device)?,
                weights.load_linear_with_optional_bias(&naming.attn_k(layer_idx), layer_device)?,
                weights.load_linear_with_optional_bias(&naming.attn_v(layer_idx), layer_device)?,
            )
        } else {
            (
                weights.load_linear(&naming.attn_q(layer_idx), layer_device)?,
                weights.load_linear(&naming.attn_k(layer_idx), layer_device)?,
                weights.load_linear(&naming.attn_v(layer_idx), layer_device)?,
            )
        };
        let o_proj = weights.load_linear(&naming.attn_output(layer_idx), layer_device)?;
        let gate_proj = weights.load_linear(&naming.ffn_gate(layer_idx), layer_device)?;
        let up_proj = weights.load_linear(&naming.ffn_up(layer_idx), layer_device)?;
        let down_proj = weights.load_linear(&naming.ffn_down(layer_idx), layer_device)?;
        let attn_norm =
            weights.load_rms_norm(&naming.attn_norm(layer_idx), config.rms_norm_eps, layer_device)?;
        let ffn_norm =
            weights.load_rms_norm(&naming.ffn_norm(layer_idx), config.rms_norm_eps, layer_device)?;

        // Build layer config
        let layer_config = LayerConfig::new(
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            config.hidden_act,
        );
        let layer_config = if let Some(window) = config.sliding_window {
            layer_config.with_sliding_window(window)
        } else {
            layer_config
        };

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
            .rope(rotary as Arc<dyn PositionEncoding>)
            .with_attn_dtype(dtype);

        // Add paged attention if enabled
        if let AttentionImplementation::PagedAttention = attention_mechanism {
            builder =
                builder.with_paged_attn(PagedAttention::new(config.head_dim, layer_device, None)?);
        }

        // Apply model-specific customization
        let ctx = LayerCustomizerContext {
            layer_idx,
            device: layer_device,
            rms_norm_eps: config.rms_norm_eps,
        };
        builder = customizer(ctx, builder, weights)?;

        // Build the layer
        layers.push(builder.build()?);
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_config() {
        let config = LayerConfig::new(32, 8, 128, Activation::Silu)
            .with_sliding_window(4096)
            .with_softcap(50.0);

        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.softcap, Some(50.0));
    }
}
