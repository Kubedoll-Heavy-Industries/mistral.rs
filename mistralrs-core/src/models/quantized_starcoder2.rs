#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! StarCoder2 model using the generic transformer infrastructure.
//!
//! Uses `StarCoder2Block` (LayerNorm + CausalAttention + NonGatedMlp) composition
//! with GELU activation (GeluPytorchTanh).
//!
//! Supports loading from both GGUF and safetensors formats via `FromGGUF` and
//! `FromSafetensors` traits.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};

use crate::attention::PositionEncoding;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    Activation, AttentionConfig, CausalAttention, NonGatedMlp, RotaryEmbedding, TransformerBlock,
};
use crate::models::{
    standard_embed, standard_lm_head, standard_transform, LanguageModel, LanguageModelConfig,
    LanguageModelExt, Model, TransformContext, TransformerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::loaders::{
    GgufNaming, GgufWeightSource, SafetensorsNaming, SafetensorsWeightSource, TensorNaming,
    WeightSource,
};
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, false);

/// Configuration for StarCoder2 model loaded from safetensors.
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
    pub norm_epsilon: f64,
    pub hidden_act: Activation,
    pub use_bias: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
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
        self.norm_epsilon
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

/// A transformer block for StarCoder2 using pre-norm architecture.
///
/// Uses the generic `TransformerBlock` with:
/// - `LayerNorm` for normalization
/// - `CausalAttention` for attention (with RoPE)
/// - `NonGatedMlp` for feed-forward (GELU, non-gated)
type StarCoder2Block = TransformerBlock<LayerNorm, CausalAttention, NonGatedMlp>;

/// StarCoder2 model weights using the generic transformer builder infrastructure.
///
/// The model uses pre-norm architecture with:
/// - LayerNorm normalization for attention and FFN
/// - Non-gated MLP with GELU activation
/// - RoPE positional embeddings
/// - Optional biases on all projections
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
        // Log quantization info if present
        if let Some(quant_cfg) = cfg.quantization_config.as_ref() {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }

        let vb_m = vb.pp("model");
        let naming = SafetensorsNaming;

        // Create weight source for loading
        let mut weights = SafetensorsWeightSource::new(&vb_m, cfg.quantization_config.as_ref());

        // Load embedding weights
        let tok_embeddings = crate::layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &cfg.quantization_config,
        )?;

        // Load output norm (LayerNorm for StarCoder2)
        let output_norm = weights.load_layer_norm(&naming.output_norm(), cfg.norm_epsilon, device)?;

        // Load output weights (may be tied to embeddings)
        let output: Arc<dyn QuantMethod> = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear(
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

        // Determine layer range for pipeline parallelism
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(cfg.num_hidden_layers))
            .unwrap_or(cfg.num_hidden_layers);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < cfg.num_hidden_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                cfg.num_hidden_layers
            );
        }

        let head_dim = cfg.head_dim();

        // Create RoPE embeddings per device
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    layer_device,
                    true,
                    dtype,
                )?));
            }
        }

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_loaded_layers);
        let vb_l = vb_m.pp("layers");
        let quant_cfg = cfg.quantization_config.clone();

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
            let vb_attn = vb_layer.pp("self_attn");
            let vb_mlp = vb_layer.pp("mlp");

            // Load attention projections with biases (StarCoder2 uses biases)
            let q_proj = mistralrs_quant::linear(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("q_proj"),
            )?;
            let k_proj = mistralrs_quant::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("k_proj"),
            )?;
            let v_proj = mistralrs_quant::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                &quant_cfg,
                vb_attn.pp("v_proj"),
            )?;
            let o_proj = mistralrs_quant::linear(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                &quant_cfg,
                vb_attn.pp("o_proj"),
            )?;

            // Load MLP weights (StarCoder2 uses c_fc/c_proj naming, not gate/up)
            let up_proj = mistralrs_quant::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                &quant_cfg,
                vb_mlp.pp("c_fc"),
            )?;
            let down_proj = mistralrs_quant::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                &quant_cfg,
                vb_mlp.pp("c_proj"),
            )?;
            let mlp = NonGatedMlp::from_weights(up_proj, down_proj, Activation::GeluPytorchTanh);

            // Load LayerNorms
            let attn_norm = crate::layers::layer_norm(
                cfg.hidden_size,
                cfg.norm_epsilon,
                vb_layer.pp("input_layernorm"),
            )?;
            let ffn_norm = crate::layers::layer_norm(
                cfg.hidden_size,
                cfg.norm_epsilon,
                vb_layer.pp("post_attention_layernorm"),
            )?;

            // Build CausalAttention
            let attn_config = AttentionConfig::new(cfg.num_attention_heads, cfg.num_key_value_heads, head_dim);
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
            max_seq_len: cfg.max_position_embeddings,
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
    ) -> Result<Self> {
        // Verify architecture
        let meta = ct.get_metadata();
        let arch: String = {
            use crate::utils::gguf_metadata::TryValueInto;
            meta.get("general.architecture")
                .cloned()
                .try_value_into()?
        };
        if arch != "starcoder2" {
            candle_core::bail!("Expected `starcoder2` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };

        // StarCoder2 uses attention.layer_norm_epsilon (not rms_epsilon)
        // So we need to extract it manually since TransformerConfig expects rms_epsilon
        let head_count = metadata
            .get_value::<u32>("attention.head_count")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let head_count_kv = metadata
            .get_value::<u32>("attention.head_count_kv")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let block_count = metadata
            .get_value::<u32>("block_count")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let embedding_length = metadata
            .get_value::<u32>("embedding_length")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let intermediate_size = metadata
            .get_value::<u32>("feed_forward_length")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let layer_norm_epsilon = metadata
            .get_value::<f32>("attention.layer_norm_epsilon")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as f64;
        let context_window = metadata
            .get_value::<u32>("context_length")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))? as usize;
        let rope_freq_base = metadata.get_value("rope.freq_base").ok().unwrap_or(100_000_f32);

        let head_dim = embedding_length / head_count;

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
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(layer_device.location())
            {
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
            let q_proj = weights.load_linear_with_optional_bias(
                &naming.attn_q(layer_idx),
                embedding_length,
                q_out_dim,
                layer_device,
            )?;
            let k_proj = weights.load_linear_with_optional_bias(
                &naming.attn_k(layer_idx),
                embedding_length,
                kv_out_dim,
                layer_device,
            )?;
            let v_proj = weights.load_linear_with_optional_bias(
                &naming.attn_v(layer_idx),
                embedding_length,
                kv_out_dim,
                layer_device,
            )?;
            let o_proj = weights.load_linear_with_optional_bias(
                &naming.attn_output(layer_idx),
                q_out_dim,
                embedding_length,
                layer_device,
            )?;

            // Load MLP weights with optional biases
            let up_proj = weights.load_linear_with_optional_bias(
                &naming.ffn_up(layer_idx),
                embedding_length,
                intermediate_size,
                layer_device,
            )?;
            let down_proj = weights.load_linear_with_optional_bias(
                &naming.ffn_down(layer_idx),
                intermediate_size,
                embedding_length,
                layer_device,
            )?;
            let mlp = NonGatedMlp::from_weights(up_proj, down_proj, Activation::GeluPytorchTanh);

            // Load LayerNorms (Starcoder2 uses LayerNorm, not RmsNorm)
            let attn_norm = weights.load_layer_norm(
                &naming.attn_norm(layer_idx).replace(".weight", ""),
                layer_norm_epsilon,
                layer_device,
            )?;
            let ffn_norm = weights.load_layer_norm(
                &naming.ffn_norm(layer_idx).replace(".weight", ""),
                layer_norm_epsilon,
                layer_device,
            )?;

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
