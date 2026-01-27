#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3 MoE model with unified loading from GGUF and safetensors formats.
//!
//! This model supports both quantized GGUF loading with optimized expert routing
//! and safetensors loading with the generic MoE infrastructure.
//!
//! # Architecture
//!
//! Qwen3 MoE uses:
//! - RMS normalization for attention and FFN
//! - Q/K normalization (always present)
//! - Mixture of Experts with SoftmaxTopK routing
//! - Some layers are dense MLP, others are MoE (controlled by decoder_sparse_step)

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{self, Activation, CausalMasker, FeedForward, MatMul, RmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::{LanguageModel, Model, TransformContext, TransformerModel};
use crate::moe::routing::{RoutingConfig, SoftmaxTopK};
use crate::moe::{LoadedExpertWeights, MoE, MoEExperts, MoELayerConfig, QuantProperties};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{ColumnParallelLayer, GgufMatMul, QuantMethod, QuantMethodConfig, QuantizedConfig, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder};

// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

// =============================================================================
// Safetensors Configuration (config.json)
// =============================================================================

serde_default_fn!(bool, tie_word_embeddings, false);

/// Configuration for Qwen3 MoE model loaded from safetensors.
/// Mirrors the config.json structure from HuggingFace.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub mlp_only_layers: Vec<usize>,
    pub decoder_sparse_step: usize,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Create MoE layer config from model config.
    fn to_moe_config(&self) -> MoELayerConfig {
        MoELayerConfig {
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: 1.0, // Qwen3 uses normalized top-k
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
        }
    }

    /// Check if a layer should be MoE based on layer index.
    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && (self.num_experts > 0 && (layer_idx + 1) % self.decoder_sparse_step == 0)
    }
}


/// Unified FFN layer for Qwen3 MoE models.
///
/// Both GGUF and safetensors loading use `MoE<SoftmaxTopK>` for MoE layers.
/// The `QuantizedMlp` variant is used for dense MLP layers in GGUF models.
enum Qwen3MoeFfn {
    /// Unified MoE layer (GGUF and safetensors)
    MoE(MoE<SoftmaxTopK>),
    /// Pre-quantized dense MLP (GGUF)
    QuantizedMlp(QuantizedMlp),
    /// Standard dense MLP (safetensors)
    Mlp(layers::Mlp),
}

impl FeedForward for Qwen3MoeFfn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE(m) => m.forward(xs),
            Self::QuantizedMlp(m) => m.forward(xs),
            Self::Mlp(m) => m.forward(xs),
        }
    }
}

impl Qwen3MoeFfn {
    /// Get ISQ layers from the MoE for quantization.
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            Self::MoE(moe) => moe.get_isq_layers(),
            Self::Mlp(mlp) => vec![&mut mlp.gate, &mut mlp.up, &mut mlp.down],
            // GGUF layers are already quantized
            Self::QuantizedMlp(_) => vec![],
        }
    }
}

/// GGUF-format MLP using quantized weights.
struct QuantizedMlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl QuantizedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;
        let y = crate::ops::mul_and_act(&w1, &w3, Activation::Silu)?;
        MatMul.qmethod_matmul(&y, &*self.feed_forward_w2)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ffn: Qwen3MoeFfn,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = MatMul.qmethod_matmul(x, &*self.attention_wq)?;
        let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?;
        let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?;

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Per-head RMSNorm in Qwen3
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let y = MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    /// Final norm layer (None for non-last pipeline stages)
    norm: Option<RmsNorm>,
    /// Output/LM head layer (None for non-last pipeline stages)
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QwenMoEConfig {
    pub moe_intermediate_size: usize,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
}

pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub moe_cfg: QwenMoEConfig,
}

fn verify_qwen3_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen3" && actual_arch != "qwen3moe" {
        candle_core::bail!("Expected `qwen3` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen3_arch(c.metadata)?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            num_experts: Some(c.get_value::<u32>("expert_count")? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
        };

        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
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
            moe_cfg,
        };

        Ok(props)
    }
}

/// Load expert weights from GGUF into LoadedExpertWeights.
///
/// This function loads expert weights and returns them in a format suitable for
/// `MoEExperts::from_loaded()`, enabling unified backend selection.
fn load_gguf_experts<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    moe_cfg: &QwenMoEConfig,
    device: &Device,
) -> Result<LoadedExpertWeights> {
    use candle_core::quantized::QTensor;

    let num_experts = moe_cfg.num_experts.unwrap();
    let mut gate_proj = Vec::with_capacity(num_experts);
    let mut up_proj = Vec::with_capacity(num_experts);
    let mut down_proj = Vec::with_capacity(num_experts);

    // Qwen3 MoE uses stacked expert format
    let gate_exps = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
    let down_exps = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
    let up_exps = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;

    let gate_type = gate_exps.dtype();
    let down_type = down_exps.dtype();
    let up_type = up_exps.dtype();
    let detected_dtype = gate_type;

    let gate_chunks = gate_exps.dequantize(device)?.chunk(num_experts, 0)?;
    let down_chunks = down_exps.dequantize(device)?.chunk(num_experts, 0)?;
    let up_chunks = up_exps.dequantize(device)?.chunk(num_experts, 0)?;

    for ((gate, down), up) in gate_chunks
        .into_iter()
        .zip(down_chunks)
        .zip(up_chunks)
    {
        gate_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(QTensor::quantize(&gate, gate_type)?),
            b: None,
        })?) as Arc<dyn QuantMethod>);
        up_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(QTensor::quantize(&up, up_type)?),
            b: None,
        })?) as Arc<dyn QuantMethod>);
        down_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(QTensor::quantize(&down, down_type)?),
            b: None,
        })?) as Arc<dyn QuantMethod>);
    }

    // Create quant properties for backend selection
    let quant_properties = QuantProperties::gguf(detected_dtype);

    Ok(LoadedExpertWeights {
        gate_proj,
        up_proj,
        down_proj,
        quant_properties,
    })
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
        // Parameter extraction from metadata.
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen3_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
            moe_cfg,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for partial loading (pipeline parallelism)
        let total_layers = block_count;
        let layer_range = layer_range.unwrap_or(0..total_layers);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(total_layers);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < total_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                total_layers
            );
        }

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;

        // PP: Only load norm and output (LM head) for last stage
        let is_last_stage = layer_end >= total_layers;
        let norm = if is_last_stage {
            Some(RmsNorm::from_qtensor(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?)
        } else {
            None
        };
        let output = if is_last_stage {
            Some(if !ct.has_tensor("output.weight") {
                ct.tensor("token_embd.weight", device)?
            } else {
                ct.tensor("output.weight", device)?
            })
        } else {
            None
        };
        let mut layers = Vec::with_capacity(num_loaded_layers);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        // Only create RoPE embeddings for loaded layers
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?),
            );
        }

        // Only load layers in the specified range
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

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

            let ffn_layer = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap()
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap() > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap() == 0)
            {
                // Load MoE gate
                let gate = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?),
                    b: None,
                })?) as Arc<dyn QuantMethod>;

                // Load experts into LoadedExpertWeights
                let loaded_experts = load_gguf_experts(&mut ct, &prefix, &moe_cfg, device)?;

                // Get comm for tensor parallelism
                let comm = mapper.get_comm_for(layer_idx)?;

                // Create unified MoEExperts with automatic backend selection
                let moe_experts = MoEExperts::from_loaded(
                    loaded_experts,
                    moe_cfg.num_experts_per_tok,
                    &comm,
                    Activation::Silu,
                )?;

                // Create unified MoE layer
                let routing_config = if moe_cfg.norm_topk_prob {
                    RoutingConfig::new_normalized(
                        moe_cfg.num_experts.unwrap(),
                        moe_cfg.num_experts_per_tok,
                    )
                } else {
                    RoutingConfig::new_scaled(
                        moe_cfg.num_experts.unwrap(),
                        moe_cfg.num_experts_per_tok,
                        1.0,
                    )
                };
                let moe = MoE::from_parts(gate, moe_experts, routing_config);

                Qwen3MoeFfn::MoE(moe)
            } else {
                let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                let mlp = QuantizedMlp {
                    feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w1),
                        b: None,
                    })?),
                    feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w2),
                        b: None,
                    })?),
                    feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w3),
                        b: None,
                    })?),
                };
                Qwen3MoeFfn::QuantizedMlp(mlp)
            };

            // Qwen3 always has q_norm and k_norm
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                q_norm,
                k_norm,
                ffn: ffn_layer,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: output.map(|q_tensor| {
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(q_tensor),
                    b: None,
                }).unwrap()) as Arc<dyn QuantMethod>
            }),
            device: device.clone(),
            max_seq_len,
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
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }

        // Determine layer range for partial loading (pipeline parallelism)
        let total_layers = cfg.num_hidden_layers;
        let layer_range = layer_range.unwrap_or(0..total_layers);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(total_layers);

        if layer_start > 0 || layer_end < total_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                total_layers
            );
        }

        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");

        // Load embeddings
        let embed_tokens = crate::layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        // PP: Only load norm and output (LM head) for last stage
        let is_last_stage = layer_end >= total_layers;
        let norm = if is_last_stage {
            Some(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_nm_device(vb_m.pp("norm"), false),
            )?)
        } else {
            None
        };

        let output = if is_last_stage {
            Some(if !cfg.tie_word_embeddings {
                ReplicatedLayer::new(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    &cfg.quantization_config,
                    false,
                    mapper.set_nm_device(vb_lm_head, false),
                )?
            } else {
                ReplicatedLayer::from_linear(candle_nn::Linear::new(
                    mapper.cast_nm_device(embed_tokens.embeddings(), false)?,
                    None,
                ))?
            })
        } else {
            None
        };

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                layer_device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    layer_device,
                    true,
                    vb.dtype(),
                )?),
            );
        }

        let comm = mapper.get_comm_for(0)?;
        let vb_l = vb_m.pp("layers");

        // Load transformer layers
        let load_in_parallel = !(device.is_metal() && cfg.quantization_config.is_none());
        let layers: Vec<LayerWeights> = NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        )
        .run(load_in_parallel, |layer_idx| {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&layer_device.location())
                .expect("No RoPE for device location!")
                .clone();

            let vb_layer = vb_l.pp(layer_idx);
            let vb_attn = mapper.set_device(layer_idx, vb_layer.pp("self_attn"), false);
            let vb_mlp = mapper.set_device(layer_idx, vb_layer.pp("mlp"), false);

            // Attention projections
            let attention_wq = ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &cfg.quantization_config,
                false,
                &comm,
                vb_attn.pp("q_proj"),
            )?;
            let kv_shard = mistralrs_quant::compute_kv_shard(
                cfg.num_key_value_heads,
                head_dim,
                &comm,
            );
            let attention_wk = ColumnParallelLayer::new_with_shard(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                &cfg.quantization_config,
                false,
                &comm,
                kv_shard,
                vb_attn.pp("k_proj"),
            )?;
            let attention_wv = ColumnParallelLayer::new_with_shard(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                &cfg.quantization_config,
                false,
                &comm,
                kv_shard,
                vb_attn.pp("v_proj"),
            )?;
            let attention_wo = RowParallelLayer::new(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                &comm,
                vb_attn.pp("o_proj"),
            )?;

            // Q/K normalization (always present in Qwen3)
            let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb_attn.pp("q_norm"))?;
            let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb_attn.pp("k_norm"))?;

            // FFN layer (MoE or MLP)
            let ffn_layer = if cfg.is_moe_layer(layer_idx) {
                // MoE layer using the type-safe MoE<SoftmaxTopK>
                let moe = MoE::<SoftmaxTopK>::new(
                    &cfg.to_moe_config(),
                    vb_mlp,
                    layer_device.clone(),
                    &comm,
                    false,
                )?;
                Qwen3MoeFfn::MoE(moe)
            } else {
                // Dense MLP layer
                let mlp = layers::Mlp::new(
                    vb_mlp,
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    &cfg.quantization_config,
                    cfg.hidden_act,
                    &comm,
                )?;
                Qwen3MoeFfn::Mlp(mlp)
            };

            // Norms
            let attention_norm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("input_layernorm"), false),
            )?;
            let ffn_norm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_layer.pp("post_attention_layernorm"), false),
            )?;

            // Paged attention
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, layer_device, None)?)
                }
            };

            Ok(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_wo,
                attention_norm,
                q_norm,
                k_norm,
                ffn: ffn_layer,
                ffn_norm,
                n_head: cfg.num_attention_heads / comm.world_size(),
                n_kv_head: (cfg.num_key_value_heads / comm.world_size()).max(1),
                head_dim,
                rotary,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                        cfg.num_key_value_heads,
                        cfg.num_attention_heads,
                        &comm,
                    ),
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                },
                dtype,
            })
        })?;

        Ok(Self {
            tok_embeddings: embed_tokens,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: cfg.max_position_embeddings,
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

            let residual = &hidden;
            let x = layer.attention_norm.forward(&hidden)?;
            let attn = layer.forward_attn(
                &x,
                mask.map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                position_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
            )?;
            let x = (attn + residual)?;

            // MLP (with MoE routing for sparse layers)
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.ffn.forward(&x)?;
            hidden = (x + residual)?;
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
