#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Gemma 2 model using a custom transformer layer for interleaved attention.
//!
//! Gemma 2 has several unique features that require a custom layer implementation:
//! - **Interleaved attention**: Even layers use sliding window, odd layers use global attention
//! - **Soft-capping**: Attention logits are capped using `cap * tanh(logits / cap)`
//! - **Post-norms**: Additional normalization after attention and FFN (4 norms per layer)
//! - **RmsNorm with +1 offset**: Like Gemma 1, weights are `1 + learned_weight`
//!
//! This file implements a custom `Gemma2Layer` that handles these features, rather than
//! using the generic `TransformerBlock`.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantMethodConfig, QuantizedConfig, RowParallelLayer,
    ShardedVarBuilder, UnquantLinear,
};

use crate::attention::SdpaParams;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    embedding, Activation, CausalMasker, MatMul, Mlp, RmsNorm, RotaryEmbedding, Sdpa,
};
use crate::models::{
    LanguageModel, Model, PagedAttentionContext, TransformContext, TokenizerModel,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::loaders::{
    GgufNaming, GgufWeightSource, TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, word_emb_default, false);

/// Configuration for Gemma 2 model loaded from safetensors.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_act: Option<Activation>,
    pub hidden_activation: Option<Activation>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn get_hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(act), Some(_)) => Ok(act),
            (None, None) => candle_core::bail!("none of hidden_act and hidden_activation are set"),
        }
    }
}

// =============================================================================
// Gemma 2 Custom Attention
// =============================================================================

/// Gemma 2 attention with soft-capping and interleaved sliding window support.
struct Gemma2Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    rotary_emb: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Whether this layer uses sliding window attention (even layers)
    use_sliding_window: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Gemma2Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        // Select mask based on whether this layer uses sliding window
        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    flash_params,
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        flash_params,
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, flash_params, &self.sdpa_params)?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// =============================================================================
// Gemma 2 Custom Layer
// =============================================================================

/// Gemma 2 decoder layer with 4 norms (pre+post attention, pre+post FFN).
struct Gemma2Layer {
    self_attn: Gemma2Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl Gemma2Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        // Pre-norm attention + post-attention norm
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(
                &xs,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                kv_cache,
                metadata,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;

        // Pre-norm FFN + post-FFN norm
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

// =============================================================================
// Gemma 2 Weight Loading Extension
// =============================================================================

/// Extension trait for loading Gemma-style RmsNorm weights (with +1 offset).
trait GemmaWeightSource: WeightSource {
    fn load_gemma_rms_norm(&mut self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm>;
}

impl<R: std::io::Seek + std::io::Read> GemmaWeightSource for GgufWeightSource<'_, '_, R> {
    fn load_gemma_rms_norm(&mut self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm> {
        let norm = self.load_rms_norm(name, eps, device)?;
        let weight = (norm.weight() + 1.0)?;
        Ok(RmsNorm::from_weight(weight, eps))
    }
}

// =============================================================================
// Model Implementation
// =============================================================================

/// Gemma 2 model weights with custom layer structure.
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Gemma2Layer>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    hidden_size: usize,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    num_attn_heads: usize,
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
        _adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        let hidden_act = cfg.get_hidden_act()?;

        let vb_m = vb.pp("model");

        let tok_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(cfg.num_hidden_layers))
            .unwrap_or(cfg.num_hidden_layers);

        // Create RoPE per device
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    layer_device,
                    true,
                    dtype,
                )?));
            }
        }

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(layer_end - layer_start);

        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary_emb = ropes.get(&layer_device.location()).unwrap().clone();
            let comm = mapper.get_comm_for(layer_idx)?;
            let vb_layer = vb_l.pp(layer_idx);

            let use_sliding_window = layer_idx % 2 == 0;
            let layer_sliding_window = if use_sliding_window {
                Some(cfg.sliding_window)
            } else {
                None
            };

            // Load attention projections
            let vb_attn = mapper.set_device(layer_idx, vb_layer.pp("self_attn"), false);
            let q_proj = ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                &cfg.quantization_config,
                cfg.attention_bias,
                &comm,
                vb_attn.pp("q_proj"),
            )?;
            let kv_shard = mistralrs_quant::compute_kv_shard(
                cfg.num_key_value_heads,
                cfg.head_dim,
                &comm,
            );
            let k_proj = ColumnParallelLayer::new_with_shard(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                &cfg.quantization_config,
                cfg.attention_bias,
                &comm,
                kv_shard,
                vb_attn.pp("k_proj"),
            )?;
            let v_proj = ColumnParallelLayer::new_with_shard(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                &cfg.quantization_config,
                cfg.attention_bias,
                &comm,
                kv_shard,
                vb_attn.pp("v_proj"),
            )?;
            let o_proj = RowParallelLayer::new(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                cfg.attention_bias,
                &comm,
                vb_attn.pp("o_proj"),
            )?;

            let num_heads = cfg.num_attention_heads / comm.world_size();
            let num_kv_heads = (cfg.num_key_value_heads / comm.world_size()).max(1);

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, layer_device, None)?)
                }
            };

            let attention = Gemma2Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rotary_emb,
                num_heads,
                num_kv_heads,
                head_dim: cfg.head_dim,
                use_sliding_window,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                        cfg.num_key_value_heads,
                        cfg.num_attention_heads,
                        &comm,
                    ),
                    softcap: cfg.attn_logit_softcapping.map(|x| x as f32),
                    softmax_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
                    sliding_window: layer_sliding_window,
                },
            };

            // Load MLP
            let mlp = Mlp::new(
                mapper.set_device(layer_idx, vb_layer.pp("mlp"), false),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                hidden_act,
                &comm,
            )?;

            // Load all 4 norms (Gemma-style with +1 offset)
            let input_layernorm = RmsNorm::new_gemma(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = RmsNorm::new_gemma(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("post_attention_layernorm"), false),
            )?;
            let pre_feedforward_layernorm = RmsNorm::new_gemma(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("pre_feedforward_layernorm"), false),
            )?;
            let post_feedforward_layernorm = RmsNorm::new_gemma(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("post_feedforward_layernorm"), false),
            )?;

            layers.push(Gemma2Layer {
                self_attn: attention,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
            });
        }

        let norm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        // Tied embeddings: lm_head shares weights with embed_tokens
        let lm_head = if cfg.tie_word_embeddings {
            let lm_head_weight = mapper.cast_nm_device(tok_embeddings.embeddings(), false)?;
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(lm_head_weight, None),
            ))?) as Arc<dyn QuantMethod>
        } else {
            ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                &mapper.get_comm_for(0)?,
                vb.pp("lm_head"),
            )?
        };

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output: lm_head,
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            mapper: Some(mapper),
            dtype,
            num_attn_heads: cfg.num_attention_heads,
            kv_dim: cfg.head_dim * cfg.num_key_value_heads,
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
        _adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        // Verify architecture
        let meta = ct.get_metadata();
        let arch: String = {
            use crate::utils::gguf_metadata::TryValueInto;
            meta.get("general.architecture").cloned().try_value_into()?
        };
        if arch != "gemma2" {
            candle_core::bail!("Expected `gemma2` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Extract Gemma 2 specific config
        let sliding_window = metadata
            .get_value::<u32>("attention.sliding_window")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(4096);

        let attn_logit_softcapping = metadata
            .get_value::<f32>("attn_logit_softcapping")
            .ok();

        let final_logit_softcapping = metadata
            .get_value::<f32>("final_logit_softcapping")
            .ok()
            .map(|x| x as f64);

        let query_pre_attn_scalar = metadata
            .get_value::<u32>("attention.query_pre_attn_scalar")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(config.head_dim);

        // Create weight source
        let mut weights = GgufWeightSource::new(&mut ct);
        let naming = GgufNaming;

        // Load embeddings
        let tok_embeddings = weights.load_embedding(
            &naming.token_embd(),
            config.vocab_size,
            config.hidden_size,
            device,
        )?;

        // Load output norm (Gemma-style with +1 offset)
        let norm = weights.load_gemma_rms_norm(&naming.output_norm(), config.rms_norm_eps, device)?;

        // Load output weights
        let output = if weights.has_tensor(&naming.output()) {
            weights.load_linear(&naming.output(), config.hidden_size, config.vocab_size, device)?
        } else {
            weights.load_linear(&naming.token_embd(), config.hidden_size, config.vocab_size, device)?
        };

        // Determine layer range
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(config.num_layers))
            .unwrap_or(config.num_layers);

        // Create RoPE per device
        let mut ropes = std::collections::HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    config.rope_theta,
                    config.head_dim,
                    config.max_seq_len,
                    layer_device,
                    true,
                    DType::F32,
                )?));
            }
        }

        // Load layers
        let mut layers = Vec::with_capacity(layer_end - layer_start);

        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary_emb = ropes.get(&layer_device.location()).unwrap().clone();

            // Determine if this layer uses sliding window (even layers)
            let use_sliding_window = layer_idx % 2 == 0;
            let layer_sliding_window = if use_sliding_window {
                Some(sliding_window)
            } else {
                None
            };

            // Load attention projections
            let q_proj = weights.load_linear(
                &naming.attn_q(layer_idx),
                config.hidden_size,
                config.num_heads * config.head_dim,
                layer_device,
            )?;
            let k_proj = weights.load_linear(
                &naming.attn_k(layer_idx),
                config.hidden_size,
                config.num_kv_heads * config.head_dim,
                layer_device,
            )?;
            let v_proj = weights.load_linear(
                &naming.attn_v(layer_idx),
                config.hidden_size,
                config.num_kv_heads * config.head_dim,
                layer_device,
            )?;
            let o_proj = weights.load_linear(
                &naming.attn_output(layer_idx),
                config.num_heads * config.head_dim,
                config.hidden_size,
                layer_device,
            )?;

            // Load MLP
            let gate = weights.load_linear(
                &naming.ffn_gate(layer_idx),
                config.hidden_size,
                config.intermediate_size,
                layer_device,
            )?;
            let up = weights.load_linear(
                &naming.ffn_up(layer_idx),
                config.hidden_size,
                config.intermediate_size,
                layer_device,
            )?;
            let down = weights.load_linear(
                &naming.ffn_down(layer_idx),
                config.intermediate_size,
                config.hidden_size,
                layer_device,
            )?;
            let mlp = Mlp::from_weights(gate, up, down, Activation::Gelu);

            // Load all 4 norms (Gemma-style with +1 offset)
            let input_layernorm = weights.load_gemma_rms_norm(
                &naming.attn_norm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;
            let post_attention_layernorm = weights.load_gemma_rms_norm(
                &format!("blk.{layer_idx}.post_attention_norm.weight"),
                config.rms_norm_eps,
                layer_device,
            )?;
            let pre_feedforward_layernorm = weights.load_gemma_rms_norm(
                &naming.ffn_norm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;
            let post_feedforward_layernorm = weights.load_gemma_rms_norm(
                &format!("blk.{layer_idx}.post_ffw_norm.weight"),
                config.rms_norm_eps,
                layer_device,
            )?;

            // Create paged attention if needed
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(config.head_dim, layer_device, None)?)
                }
            };

            let attention = Gemma2Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                rotary_emb,
                num_heads: config.num_heads,
                num_kv_heads: config.num_kv_heads,
                head_dim: config.head_dim,
                use_sliding_window,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: config.num_heads / config.num_kv_heads,
                    softcap: attn_logit_softcapping,
                    softmax_scale: 1.0 / (query_pre_attn_scalar as f32).sqrt(),
                    sliding_window: layer_sliding_window,
                },
            };

            layers.push(Gemma2Layer {
                self_attn: attention,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            hidden_size: config.hidden_size,
            sliding_window,
            final_logit_softcapping,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            mapper: Some(mapper),
            dtype,
            num_attn_heads: config.num_heads,
            kv_dim: config.head_dim * config.num_kv_heads,
        })
    }
}

impl ModelWeights {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        cache: &mut [KvCache],
        paged_attn_ctx: Option<&PagedAttentionContext>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        // Embed with Gemma scaling
        let xs = self.tok_embeddings.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        let (_b_sz, seq_len) = input_ids.dims2()?;
        let past_kv_len = seqlen_offsets.first().copied().unwrap_or(0);

        // Create masks using the newer explicit parameter functions
        let attention_mask = CausalMasker.make_causal_mask(
            seq_len,
            input_ids.device(),
            past_kv_len,
            xs.dtype(),
        )?;
        let attention_mask = attention_mask.filter(|_| {
            paged_attn_ctx
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        let sliding_attention_mask = CausalMasker.make_sliding_window_mask(
            seq_len,
            input_ids.device(),
            past_kv_len,
            self.sliding_window,
            xs.dtype(),
        )?;
        let sliding_attention_mask = sliding_attention_mask.filter(|_| {
            paged_attn_ctx
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Run through layers
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let metadata = paged_attn_ctx
                .as_ref()
                .map(|pa| (pa.kv_cache[i].clone(), pa.metadata));
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                sliding_attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                metadata,
                flash_params,
            )?;
        }

        // Output norm
        let xs = xs.to_device(&self.device)?;
        let mut xs = self.norm.forward(&xs)?;

        // LM head with optional final softcapping
        if let Some(t) = self.output.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut logits = MatMul.qmethod_matmul(&xs, &*self.output)?;

        // Apply final logit softcapping if present
        if let Some(cap) = self.final_logit_softcapping {
            logits = (logits / cap)?;
            logits = logits.tanh()?;
            logits = (logits * cap)?;
        }

        Ok(logits)
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
        let embeds = self.tok_embeddings.forward(tokens)?;
        embeds * (self.hidden_size as f64).sqrt()
    }

    fn transform(
        &self,
        _hidden: Tensor,
        _ctx: &TransformContext,
        _cache: &mut [KvCache],
    ) -> Result<Tensor> {
        // Gemma 2 uses custom forward due to dual masks
        candle_core::bail!("Gemma2 requires custom forward with dual masks")
    }
}

impl LanguageModel<[KvCache]> for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        let mut xs = self.norm.forward(&hidden)?;
        if let Some(t) = self.output.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut logits = MatMul.qmethod_matmul(&xs, &*self.output)?;

        if let Some(cap) = self.final_logit_softcapping {
            logits = (logits / cap)?;
            logits = logits.tanh()?;
            logits = (logits * cap)?;
        }

        Ok(logits)
    }
}
