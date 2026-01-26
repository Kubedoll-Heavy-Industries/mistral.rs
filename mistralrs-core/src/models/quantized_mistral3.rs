#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Quantized Mistral3 model implementation.
//!
//! Mistral3 differs from Llama in several key ways:
//! - Uses YaRN RoPE scaling for extended context
//! - Different rope parameters (beta_fast=32, beta_slow=1, factor=16)
//!
//! Temperature scaling: The GGUF `attention.temperature_scale` (typically 0.1) is the
//! coefficient for YaRN's mscale formula: `mscale = coefficient * ln(factor) + 1.0`.
//! This mscale is applied to cos/sin RoPE embeddings, NOT to the softmax scale.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};

use crate::attention::PositionEncoding;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{Activation, CausalMasker, MatMul, RmsNorm};
use crate::layers_masker::PastKvLenCache;
use crate::models::{LanguageModel, Model, TransformContext, TransformerModel};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::{
    load_transformer_from_safetensors, load_transformer_layers, GgufNaming, GgufWeightSource,
    StandardTransformerBlock, TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(f64, default_rope_theta, 1_000_000.0);
serde_default_fn!(f32, default_mscale, 0.1);

/// RoPE type for Mistral3 models
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    #[default]
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "yarn")]
    Yarn,
}

/// RoPE scaling parameters for Mistral3 (supports YaRN)
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: RopeType,
    pub factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub mscale: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
}

/// Configuration for Mistral3 model loaded from safetensors.
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
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub head_dim: Option<usize>,
    pub rope_scaling: Option<RopeScaling>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get YarnConfig from rope_scaling if present and type is yarn
    pub fn yarn_config(&self) -> Option<YarnConfig> {
        self.rope_scaling.as_ref().and_then(|rs| {
            if matches!(rs.rope_type, RopeType::Yarn) {
                Some(YarnConfig {
                    factor: rs.factor.unwrap_or(16.0),
                    original_max_position_embeddings: rs
                        .original_max_position_embeddings
                        .unwrap_or(16384),
                    beta_fast: rs.beta_fast.unwrap_or(32.0),
                    beta_slow: rs.beta_slow.unwrap_or(1.0),
                })
            } else {
                None
            }
        })
    }

    /// Get mscale coefficient for YaRN
    pub fn mscale_coefficient(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|rs| rs.mscale)
            .unwrap_or(0.1)
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
// YaRN RoPE Configuration and Embedding
// =============================================================================

/// YaRN RoPE scaling configuration for Mistral3
#[derive(Debug, Clone)]
pub struct YarnConfig {
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for YarnConfig {
    fn default() -> Self {
        Self {
            factor: 16.0,
            original_max_position_embeddings: 16384,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// YaRN-capable rotary embedding for Mistral3
#[derive(Debug, Clone)]
pub struct Mistral3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Precomputed: is rope_dim < head_dim?
    is_partial: bool,
    /// The rotary dimension (rope_dim)
    rot_dim: usize,
}

impl Mistral3RotaryEmbedding {
    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1)
    }

    /// Compute YaRN mscale for attention temperature scaling.
    ///
    /// The coefficient (typically 0.1) comes from the GGUF `attention.temperature_scale` field.
    /// Formula: mscale = coefficient * ln(factor) + 1.0
    fn yarn_get_mscale(factor: f32, coefficient: f32) -> f32 {
        if factor <= 1. {
            return 1.;
        }
        coefficient * factor.ln() + 1.
    }

    pub fn new_yarn(
        rope_theta: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        yarn: &YarnConfig,
        mscale_coefficient: f32,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;

        // Base frequencies
        let freq_extra: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;

        // Scaled frequencies
        let freq_inter: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (yarn.factor * rope_theta.powf(i as f32 / dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        // Find correction range based on beta parameters
        let (low, high) = Self::yarn_find_correction_range(
            yarn.beta_fast,
            yarn.beta_slow,
            dim,
            rope_theta,
            yarn.original_max_position_embeddings,
        );

        // Create interpolation mask
        let inv_freq_mask = (1. - Self::yarn_linear_ramp_mask(low, high, dim / 2, dev)?)?;

        // Interpolate between base and scaled frequencies
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        // Compute position embeddings
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        // Apply mscale (temperature scaling burned into RoPE embeddings)
        let mscale = Self::yarn_get_mscale(yarn.factor, mscale_coefficient);
        tracing::debug!(
            "YaRN mscale = {} * ln({}) + 1.0 = {}",
            mscale_coefficient,
            yarn.factor,
            mscale
        );
        let sin = (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?;
        let rot_dim = dim;

        Ok(Self { sin, cos, is_partial: false, rot_dim })
    }

    pub fn new_unscaled(
        rope_theta: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let rot_dim = dim;

        Ok(Self { sin, cos, is_partial: false, rot_dim })
    }

    /// Mark this embedding as partial rotary (rope_dim < head_dim)
    pub fn set_partial(&mut self, head_dim: usize) {
        self.is_partial = self.rot_dim < head_dim;
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, n_embd) = q.dims4()?;

        // Handle partial rotary: only apply RoPE to first rot_dim dimensions
        if self.is_partial {
            let rot_dim = self.rot_dim;
            let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
            let q_pass = q.narrow(D::Minus1, rot_dim, n_embd - rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
            let k_pass = k.narrow(D::Minus1, rot_dim, n_embd - rot_dim)?;

            let (q_rot, k_rot) = if seqlen_offsets.len() == 1 {
                // NOTE: Don't call .to_dtype() here - cos/sin are already in the target dtype
                // from initialization, and extra conversion can introduce numerical noise
                let cos = self.cos.narrow(0, seqlen_offsets[0], seq_len)?;
                let sin = self.sin.narrow(0, seqlen_offsets[0], seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope_i(&q_rot.contiguous()?, &cos, &sin)?;
                let k_embed = candle_nn::rotary_emb::rope_i(&k_rot.contiguous()?, &cos, &sin)?;
                (q_embed, k_embed)
            } else {
                let mut q_embeds = Vec::new();
                let mut k_embeds = Vec::new();
                for (i, offset) in seqlen_offsets.iter().enumerate() {
                    let cos = self.cos.narrow(0, *offset, seq_len)?;
                    let sin = self.sin.narrow(0, *offset, seq_len)?;
                    let q_embed = candle_nn::rotary_emb::rope_i(
                        &q_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    let k_embed = candle_nn::rotary_emb::rope_i(
                        &k_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    q_embeds.push(q_embed);
                    k_embeds.push(k_embed);
                }
                (Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?)
            };

            Ok((
                Tensor::cat(&[q_rot, q_pass], D::Minus1)?.contiguous()?,
                Tensor::cat(&[k_rot, k_pass], D::Minus1)?.contiguous()?,
            ))
        } else {
            // Full rotary: apply RoPE to all dimensions
            if seqlen_offsets.len() == 1 {
                // NOTE: Don't call .to_dtype() here - cos/sin are already in the target dtype
                // from initialization, and extra conversion can introduce numerical noise
                let cos = self.cos.narrow(0, seqlen_offsets[0], seq_len)?;
                let sin = self.sin.narrow(0, seqlen_offsets[0], seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
                let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
                Ok((q_embed, k_embed))
            } else {
                let mut q_embeds = Vec::new();
                let mut k_embeds = Vec::new();
                for (i, offset) in seqlen_offsets.iter().enumerate() {
                    let cos = self.cos.narrow(0, *offset, seq_len)?;
                    let sin = self.sin.narrow(0, *offset, seq_len)?;
                    let q_embed =
                        candle_nn::rotary_emb::rope_i(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                    let k_embed =
                        candle_nn::rotary_emb::rope_i(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                    q_embeds.push(q_embed);
                    k_embeds.push(k_embed);
                }
                Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
            }
        }
    }
}

/// Implement PositionEncoding trait so YaRN RoPE can be used with CausalAttention.
impl PositionEncoding for Mistral3RotaryEmbedding {
    fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offsets: &[usize]) -> Result<(Tensor, Tensor)> {
        Mistral3RotaryEmbedding::forward(self, q, k, seqlen_offsets)
    }
}

/// Mistral3 model weights using the generic transformer builder infrastructure.
///
/// Uses YaRN RoPE for extended context and standard Llama-like architecture:
/// - RMS normalization for attention and FFN
/// - Gated MLP with SiLU activation
/// - YaRN-scaled RoPE positional embeddings
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    /// Final norm layer (None for non-last pipeline stages)
    norm: Option<RmsNorm>,
    /// Output/LM head layer (None for non-last pipeline stages)
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

/// Mistral3 GGUF properties
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub yarn_config: Option<YarnConfig>,
    pub attn_temperature_scale: Option<f32>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("mistral3")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // Read YaRN scaling parameters if present
        let yarn_config = {
            let scaling_type: Option<String> = c.get_value("rope.scaling.type").ok();
            if scaling_type.as_deref() == Some("yarn") {
                Some(YarnConfig {
                    factor: c.get_value("rope.scaling.factor").unwrap_or(16.0),
                    original_max_position_embeddings: c
                        .get_value::<u64>("rope.scaling.original_context_length")
                        .unwrap_or(16384) as usize,
                    beta_fast: c.get_value("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
                    beta_slow: c.get_value("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
                })
            } else {
                None
            }
        };

        // Read attention temperature scale (Mistral3 specific)
        let attn_temperature_scale: Option<f32> = c.get_value("attention.temperature_scale").ok();

        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(1_000_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            yarn_config,
            attn_temperature_scale,
        };

        Ok(props)
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
        // Parse Mistral3-specific metadata (includes YaRN config)
        let metadata = ContentMetadata {
            path_prefix: "mistral3",
            metadata: ct.get_metadata(),
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Parse common transformer config from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: "mistral3",
            metadata: ct.get_metadata(),
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .with_head_dim(props.key_length)
            .with_rope_theta(props.rope_freq_base);

        tracing::info!(
            "Mistral3 model config: head_dim={}, rope_dim={}, head_count={}, head_count_kv={}, embedding_length={}, block_count={}",
            config.head_dim,
            props.rope_dim,
            config.num_heads,
            config.num_kv_heads,
            config.hidden_size,
            config.num_layers
        );

        // Determine layer range for pipeline parallelism
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(config.num_layers))
            .unwrap_or(config.num_layers);

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

        // PP: Only load norm and output (LM head) for last stage
        let is_last_stage = layer_end >= config.num_layers;
        let norm = if is_last_stage {
            Some(weights.load_rms_norm(&naming.output_norm(), config.rms_norm_eps, device)?)
        } else {
            None
        };
        let output = if is_last_stage {
            Some(if weights.has_tensor(&naming.output()) {
                weights.load_linear(&naming.output(), config.hidden_size, config.vocab_size, device)?
            } else {
                weights.load_linear(&naming.token_embd(), config.hidden_size, config.vocab_size, device)?
            })
        } else {
            None
        };

        // Create YaRN RoPE embeddings per device (Mistral3-specific)
        // The temperature_scale from GGUF is the coefficient for YaRN mscale formula
        let mscale_coefficient = props.attn_temperature_scale.unwrap_or(0.1);
        let mut yarn_ropes: HashMap<candle_core::DeviceLocation, Arc<Mistral3RotaryEmbedding>> =
            HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                yarn_ropes.entry(layer_device.location())
            {
                let mut rope = if let Some(ref yarn) = props.yarn_config {
                    tracing::info!(
                        "Using YaRN RoPE scaling for Mistral3: factor={}, beta_fast={}, beta_slow={}, mscale_coefficient={}, rope_dim={}",
                        yarn.factor,
                        yarn.beta_fast,
                        yarn.beta_slow,
                        mscale_coefficient,
                        props.rope_dim
                    );
                    Mistral3RotaryEmbedding::new_yarn(
                        props.rope_freq_base,
                        props.rope_dim,
                        config.max_seq_len,
                        yarn,
                        mscale_coefficient,
                        dtype,
                        layer_device,
                    )?
                } else {
                    Mistral3RotaryEmbedding::new_unscaled(
                        props.rope_freq_base,
                        props.rope_dim,
                        config.max_seq_len,
                        dtype,
                        layer_device,
                    )?
                };
                // Set partial rotary flag if rope_dim < head_dim
                rope.set_partial(config.head_dim);
                e.insert(Arc::new(rope));
            }
        }

        // Load transformer layers using generic infrastructure
        // Mistral3-specific: inject YaRN RoPE via customizer
        let layers = load_transformer_layers(
            &config,
            &mut weights,
            &naming,
            layer_range,
            &*mapper,
            device,
            attention_mechanism,
            dtype,
            |ctx, builder, _weights| {
                // Replace default RoPE with YaRN RoPE for this layer's device
                let yarn_rope = yarn_ropes
                    .get(&ctx.device.location())
                    .expect("No YaRN RoPE for device location!")
                    .clone();
                Ok(builder.rope(yarn_rope as Arc<dyn PositionEncoding>))
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
        // Derived values for YaRN RoPE setup
        let num_layers = cfg.num_hidden_layers;
        let head_dim = cfg.head_dim();

        // Determine layer range for YaRN RoPE setup
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(num_layers))
            .unwrap_or(num_layers);

        // Create YaRN RoPE embeddings per device (Mistral3-specific, BEFORE layer loading)
        let yarn_config = cfg.yarn_config();
        let mscale_coefficient = cfg.mscale_coefficient();
        let mut yarn_ropes: HashMap<candle_core::DeviceLocation, Arc<Mistral3RotaryEmbedding>> =
            HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                yarn_ropes.entry(layer_device.location())
            {
                let mut rope = if let Some(ref yarn) = yarn_config {
                    tracing::info!(
                        "Using YaRN RoPE scaling for Mistral3: factor={}, beta_fast={}, beta_slow={}, mscale_coefficient={}, head_dim={}",
                        yarn.factor,
                        yarn.beta_fast,
                        yarn.beta_slow,
                        mscale_coefficient,
                        head_dim
                    );
                    Mistral3RotaryEmbedding::new_yarn(
                        cfg.rope_theta as f32,
                        head_dim,
                        cfg.max_position_embeddings,
                        yarn,
                        mscale_coefficient,
                        dtype,
                        layer_device,
                    )?
                } else {
                    Mistral3RotaryEmbedding::new_unscaled(
                        cfg.rope_theta as f32,
                        head_dim,
                        cfg.max_position_embeddings,
                        dtype,
                        layer_device,
                    )?
                };
                // Set partial rotary flag if rope_dim < head_dim
                // For safetensors, assume full rotary (rope_dim == head_dim)
                rope.set_partial(head_dim);
                e.insert(Arc::new(rope));
            }
        }

        // Load transformer with YaRN RoPE customization
        let loaded = load_transformer_from_safetensors(
            cfg,
            TransformerConfig::from_config(cfg),
            vb,
            device,
            &*mapper,
            attention_mechanism,
            dtype,
            layer_range,
            |ctx, builder, _weights| {
                // Replace default RoPE with YaRN RoPE for this layer's device
                let yarn_rope = yarn_ropes
                    .get(&ctx.device.location())
                    .expect("No YaRN RoPE for device location!")
                    .clone();
                Ok(builder.rope(yarn_rope as Arc<dyn PositionEncoding>))
            },
        )?;

        Ok(Self {
            tok_embeddings: loaded.tok_embeddings,
            layers: loaded.layers,
            norm: Some(loaded.output_norm),
            output: Some(loaded.output),
            device: device.clone(),
            max_seq_len: loaded.max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    /// Run transformer layers on hidden states.
    ///
    /// This is the core layer iteration logic used by `transform()`.
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

            hidden = layer.forward(
                hidden,
                mask,
                position_offsets,
                &mut cache[i],
                layer_metadata,
            )?;
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

        // Compute causal mask using position offsets
        let mask = CausalMasker.make_causal_mask_as(
            seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;
        // Only apply mask on first prompt chunk (optimization for paged attention)
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Run transformer layers
        let meta_ref = ctx
            .paged_attn
            .as_ref()
            .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));
        self.run_layers(hidden, mask.as_ref(), &start_offsets, meta_ref, cache)
    }
}

impl LanguageModel for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        // Move to model device and apply final norm
        let hidden = hidden.to_device(&self.device)?;
        let x = self
            .norm
            .as_ref()
            .expect("lm_head called on non-last pipeline stage")
            .forward(&hidden)?;
        // Project to vocabulary
        MatMul.qmethod_matmul(
            &x.contiguous()?,
            &**self
                .output
                .as_ref()
                .expect("lm_head called on non-last pipeline stage"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    /// Helper to check all values are finite (no NaN/Inf)
    fn assert_finite(t: &Tensor, name: &str) {
        let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{name} contains non-finite value at index {i}: {v}"
            );
        }
    }

    /// Helper to check values are within reasonable bounds
    fn assert_bounded(t: &Tensor, name: &str, max_abs: f32) {
        let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                v.abs() <= max_abs,
                "{name} has value {v} at index {i} exceeding bound {max_abs}"
            );
        }
    }

    #[test]
    fn test_yarn_mscale_computation() {
        // Test the mscale formula: mscale = coefficient * ln(factor) + 1.0
        let mscale = Mistral3RotaryEmbedding::yarn_get_mscale(16.0, 0.1);
        let expected = 0.1 * 16.0_f32.ln() + 1.0; // ≈ 1.277
        assert!(
            (mscale - expected).abs() < 1e-6,
            "mscale mismatch: got {mscale}, expected {expected}"
        );

        // Factor <= 1 should return 1.0
        let mscale_no_scale = Mistral3RotaryEmbedding::yarn_get_mscale(1.0, 0.1);
        assert_eq!(mscale_no_scale, 1.0);

        let mscale_low = Mistral3RotaryEmbedding::yarn_get_mscale(0.5, 0.1);
        assert_eq!(mscale_low, 1.0);
    }

    #[test]
    fn test_yarn_rotary_embedding_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();

        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1, // mscale coefficient
            dtype,
            &device,
        )
        .expect("Failed to create YaRN RoPE");

        // Check cos/sin tensors have correct shape
        assert_eq!(rope.cos.dims(), &[max_seq_len, head_dim / 2]);
        assert_eq!(rope.sin.dims(), &[max_seq_len, head_dim / 2]);

        // Check all values are finite
        assert_finite(&rope.cos.to_dtype(DType::F32).unwrap(), "cos");
        assert_finite(&rope.sin.to_dtype(DType::F32).unwrap(), "sin");

        // Check values are bounded (mscale ≈ 1.277, so max should be ~1.277)
        assert_bounded(&rope.cos.to_dtype(DType::F32).unwrap(), "cos", 2.0);
        assert_bounded(&rope.sin.to_dtype(DType::F32).unwrap(), "sin", 2.0);
    }

    #[test]
    fn test_rotary_forward_offset_zero() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        // Create Q/K tensors: (batch, heads, seq_len, head_dim)
        let batch = 1;
        let n_heads = 8;
        let seq_len = 32;
        let q = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

        let (q_rot, k_rot) = rope.forward(&q, &k, &[0]).expect("RoPE forward failed");

        // Check output shapes match input
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());

        // Check all values are finite
        assert_finite(&q_rot, "q_rot");
        assert_finite(&k_rot, "k_rot");

        // Check values are bounded (input was N(0,1), RoPE shouldn't explode values)
        assert_bounded(&q_rot, "q_rot", 100.0);
        assert_bounded(&k_rot, "k_rot", 100.0);
    }

    #[test]
    fn test_rotary_forward_with_offset() {
        // This test simulates what happens during prefix cache hit:
        // - First request processes positions 0..N
        // - Second request with prefix cache hit processes positions N..M with offset=N
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        let batch = 1;
        let n_heads = 8;
        let seq_len = 16; // New tokens after prefix cache hit

        // Simulate different offsets (as if prefix cache matched different lengths)
        for offset in [10, 50, 100, 500, 1000] {
            let q =
                Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
            let k =
                Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

            let (q_rot, k_rot) = rope
                .forward(&q, &k, &[offset])
                .unwrap_or_else(|e| panic!("RoPE forward failed at offset {offset}: {e}"));

            assert_finite(&q_rot, &format!("q_rot at offset {offset}"));
            assert_finite(&k_rot, &format!("k_rot at offset {offset}"));
            assert_bounded(&q_rot, &format!("q_rot at offset {offset}"), 100.0);
            assert_bounded(&k_rot, &format!("k_rot at offset {offset}"), 100.0);
        }
    }

    #[test]
    fn test_rotary_equivalence_split_vs_full() {
        // Test that processing in two chunks with offset gives equivalent results
        // to processing all at once. This is critical for prefix caching correctness.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        let batch = 1;
        let n_heads = 8;
        let prefix_len = 20;
        let continuation_len = 10;
        let total_len = prefix_len + continuation_len;

        // Create full Q/K tensors
        let q_full =
            Tensor::randn(0f32, 1.0, (batch, n_heads, total_len, head_dim), &device).unwrap();
        let k_full =
            Tensor::randn(0f32, 1.0, (batch, n_heads, total_len, head_dim), &device).unwrap();

        // Process all at once
        let (q_full_rot, k_full_rot) = rope.forward(&q_full, &k_full, &[0]).unwrap();

        // Process in two parts
        let q_prefix = q_full.narrow(2, 0, prefix_len).unwrap();
        let k_prefix = k_full.narrow(2, 0, prefix_len).unwrap();
        let (q_prefix_rot, k_prefix_rot) = rope.forward(&q_prefix, &k_prefix, &[0]).unwrap();

        let q_cont = q_full.narrow(2, prefix_len, continuation_len).unwrap();
        let k_cont = k_full.narrow(2, prefix_len, continuation_len).unwrap();
        let (q_cont_rot, k_cont_rot) = rope.forward(&q_cont, &k_cont, &[prefix_len]).unwrap();

        // Extract corresponding parts from full result
        let q_full_prefix = q_full_rot.narrow(2, 0, prefix_len).unwrap();
        let k_full_prefix = k_full_rot.narrow(2, 0, prefix_len).unwrap();
        let q_full_cont = q_full_rot.narrow(2, prefix_len, continuation_len).unwrap();
        let k_full_cont = k_full_rot.narrow(2, prefix_len, continuation_len).unwrap();

        // Compare prefix parts
        let q_prefix_diff = (&q_prefix_rot - &q_full_prefix)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            q_prefix_diff < 1e-5,
            "Q prefix mismatch: max diff = {q_prefix_diff}"
        );

        let k_prefix_diff = (&k_prefix_rot - &k_full_prefix)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            k_prefix_diff < 1e-5,
            "K prefix mismatch: max diff = {k_prefix_diff}"
        );

        // Compare continuation parts (this is the critical test for prefix cache)
        let q_cont_diff = (&q_cont_rot - &q_full_cont)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            q_cont_diff < 1e-5,
            "Q continuation mismatch: max diff = {q_cont_diff}"
        );

        let k_cont_diff = (&k_cont_rot - &k_full_cont)
            .unwrap()
            .abs()
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            k_cont_diff < 1e-5,
            "K continuation mismatch: max diff = {k_cont_diff}"
        );
    }

    #[test]
    fn test_rotary_multiple_forward_passes() {
        // Test numerical stability across many forward passes
        // This simulates the pattern seen in production: first few requests work,
        // then subsequent ones fail
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        let batch = 1;
        let n_heads = 8;
        let seq_len = 32;

        // Simulate many requests with varying offsets
        for iteration in 0..20 {
            // Vary offset to simulate prefix cache hits at different points
            let offset = (iteration * 50) % 2000;
            let q =
                Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
            let k =
                Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

            let (q_rot, k_rot) = rope
                .forward(&q, &k, &[offset])
                .unwrap_or_else(|e| panic!("Iteration {iteration} failed at offset {offset}: {e}"));

            assert_finite(&q_rot, &format!("q_rot iteration {iteration}"));
            assert_finite(&k_rot, &format!("k_rot iteration {iteration}"));
        }
    }

    #[test]
    fn test_rotary_f16_dtype() {
        // Test with F16 dtype (used on M3 Macs)
        let device = Device::Cpu;
        let dtype = DType::F16;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        // Verify cos/sin are in F16
        assert_eq!(rope.cos.dtype(), DType::F16);
        assert_eq!(rope.sin.dtype(), DType::F16);

        // Check values are finite after conversion to F32 for inspection
        assert_finite(&rope.cos.to_dtype(DType::F32).unwrap(), "cos F16");
        assert_finite(&rope.sin.to_dtype(DType::F32).unwrap(), "sin F16");

        let batch = 1;
        let n_heads = 8;
        let seq_len = 32;

        // Q/K are typically computed in a higher dtype then converted
        let q = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        // Test with offset
        for offset in [0, 100, 500] {
            let (q_rot, k_rot) = rope
                .forward(&q, &k, &[offset])
                .unwrap_or_else(|e| panic!("F16 forward failed at offset {offset}: {e}"));

            let q_rot_f32 = q_rot.to_dtype(DType::F32).unwrap();
            let k_rot_f32 = k_rot.to_dtype(DType::F32).unwrap();

            assert_finite(&q_rot_f32, &format!("q_rot F16 at offset {offset}"));
            assert_finite(&k_rot_f32, &format!("k_rot F16 at offset {offset}"));
        }
    }

    #[test]
    fn test_rotary_partial() {
        // Test partial rotary embedding (rope_dim < head_dim)
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let rope_dim = 64; // Partial rotary
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let mut rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            rope_dim, // Use rope_dim, not head_dim
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        // Mark as partial (rope_dim < head_dim)
        rope.set_partial(head_dim);
        assert!(rope.is_partial);

        let batch = 1;
        let n_heads = 8;
        let seq_len = 32;

        // Q/K have full head_dim
        let q = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

        for offset in [0, 100, 500] {
            let (q_rot, k_rot) = rope
                .forward(&q, &k, &[offset])
                .unwrap_or_else(|e| panic!("Partial RoPE failed at offset {offset}: {e}"));

            // Output should still have full head_dim
            assert_eq!(q_rot.dims(), q.dims());
            assert_eq!(k_rot.dims(), k.dims());

            assert_finite(&q_rot, &format!("partial q_rot at offset {offset}"));
            assert_finite(&k_rot, &format!("partial k_rot at offset {offset}"));
        }
    }

    #[test]
    fn test_rotary_multiple_offsets() {
        // Test with multiple different offsets in a batch (used when batch size > 1
        // and different sequences have different prefix cache hit lengths)
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 1_000_000.0;

        let yarn = YarnConfig::default();
        let rope = Mistral3RotaryEmbedding::new_yarn(
            rope_theta,
            head_dim,
            max_seq_len,
            &yarn,
            0.1,
            dtype,
            &device,
        )
        .unwrap();

        let batch = 4;
        let n_heads = 8;
        let seq_len = 16;

        let q = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

        // Different offsets for each batch item
        let offsets = vec![0, 50, 100, 200];

        let (q_rot, k_rot) = rope
            .forward(&q, &k, &offsets)
            .expect("Multi-offset forward failed");

        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());

        assert_finite(&q_rot, "multi-offset q_rot");
        assert_finite(&k_rot, "multi-offset k_rot");
    }

    #[test]
    fn test_unscaled_rotary() {
        // Test unscaled (non-YaRN) rotary embedding
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope_theta = 10_000.0;

        let rope =
            Mistral3RotaryEmbedding::new_unscaled(rope_theta, head_dim, max_seq_len, dtype, &device)
                .unwrap();

        // Unscaled cos/sin should be in range [-1, 1]
        assert_bounded(&rope.cos, "unscaled cos", 1.01);
        assert_bounded(&rope.sin, "unscaled sin", 1.01);

        let batch = 1;
        let n_heads = 8;
        let seq_len = 32;

        let q = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

        for offset in [0, 100, 500] {
            let (q_rot, k_rot) = rope.forward(&q, &k, &[offset]).unwrap();
            assert_finite(&q_rot, &format!("unscaled q_rot at offset {offset}"));
            assert_finite(&k_rot, &format!("unscaled k_rot at offset {offset}"));
        }
    }

    #[test]
    fn test_yarn_correction_functions() {
        // Test the YaRN correction dimension and range functions
        let dim = 128;
        let base = 1_000_000.0_f32;
        let max_pos = 16384;

        // Test correction dim calculation
        let correction_dim = Mistral3RotaryEmbedding::yarn_find_correction_dim(
            1.0, // num_rot
            dim, base, max_pos,
        );
        assert!(correction_dim.is_finite(), "correction_dim is not finite");

        // Test correction range
        let (low, high) = Mistral3RotaryEmbedding::yarn_find_correction_range(
            32.0, // beta_fast
            1.0,  // beta_slow
            dim, base, max_pos,
        );
        assert!(low.is_finite(), "correction range low is not finite");
        assert!(high.is_finite(), "correction range high is not finite");
        assert!(low <= high, "correction range: low > high");
        assert!(low >= 0.0, "correction range: low < 0");
        assert!(high <= dim as f32, "correction range: high > dim");
    }
}
