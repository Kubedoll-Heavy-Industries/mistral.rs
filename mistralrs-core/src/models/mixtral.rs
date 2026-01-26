#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Mixtral model implementation (Mixture of Experts).
//!
//! Mixtral uses the Llama architecture with MoE (Mixture of Experts) replacing
//! the standard MLP. Each token is routed to top-k experts, combining their outputs.
//!
//! This is a fundamentally different compute pattern from dense models:
//! - Sparse activation: only k of n experts run per token
//! - Different memory access patterns
//! - Different parallelism strategies
//!
//! For these reasons, Mixtral is a separate type from Llama, not a hidden variant.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, QuantizedConfig};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(bool, word_emb_default, false);

/// Mixtral configuration (for safetensors loading).
///
/// This is the JSON config.json structure from HuggingFace Mixtral models.
/// For GGUF loading, configuration is extracted from file metadata instead.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_local_experts: usize,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub(crate) tie_word_embeddings: bool,
}

use crate::{
    attention::SdpaParams,
    device_map::DeviceMapper,
    gguf::Content,
    layers::{
        reshape_attn_output, reshape_for_attn, sdpa_with_cache, CausalMasker, MatMul,
        Mlp, RmsNorm, RotaryEmbedding,
    },
    layers_masker::PastKvLenCache,
    models::{LanguageModel, Model, TransformContext, TransformerModel},
    paged_attention::{AttentionImplementation, PagedAttention},
    pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
    pipeline::KvCache,
    utils::gguf_metadata::ContentMetadata,
    utils::model_config as ModelConfig,
    utils::progress::{new_multi_progress, NiceProgressBar},
};

// =============================================================================
// Sparse Mixture of Experts
// =============================================================================

/// Sparse MoE block that routes tokens to top-k experts.
///
/// Unlike dense MLP where all parameters are used for every token,
/// MoE selects a subset of experts per token, enabling larger model
/// capacity with similar compute cost.
struct SparseMoeBlock {
    /// Router/gate that produces expert scores
    gate: Arc<dyn QuantMethod>,
    /// Expert MLPs
    experts: Vec<Mlp>,
    /// Number of experts to use per token (typically 2)
    n_expert_used: usize,
}

impl SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        // Compute routing scores
        let router_logits = MatMul.qmethod_matmul(&xs, &*self.gate)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // Select top-k experts per token
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_rws = vec![vec![]; self.experts.len()];

        for (row_idx, rw) in routing_weights.iter().enumerate() {
            // Sort expert indices by routing weight (descending)
            let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));

            // Compute normalized routing weights for top-k
            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.n_expert_used) {
                let expert_idx = expert_idx as usize;
                sum_routing_weights += rw[expert_idx];
                top_x[expert_idx].push(row_idx as u32);
            }
            for &expert_idx in dst.iter().take(self.n_expert_used) {
                let expert_idx = expert_idx as usize;
                selected_rws[expert_idx].push(rw[expert_idx] / sum_routing_weights);
            }
        }

        // Compute weighted expert outputs
        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }

            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;

            // Gather tokens for this expert
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;

            // Expert forward pass
            let current_hidden = expert.forward(&current_state)?;
            let current_hidden = current_hidden.broadcast_mul(&selected_rws)?;

            // Scatter-add back to output
            ys = ys.index_add(&top_x, &current_hidden, 0)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }
}

// =============================================================================
// Attention
// =============================================================================

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // QKV projections
        let q = MatMul
            .qmethod_matmul(x, &*self.q_proj)?
            .to_dtype(self.dtype)?;
        let k = MatMul
            .qmethod_matmul(x, &*self.k_proj)?
            .to_dtype(self.dtype)?;
        let v = MatMul
            .qmethod_matmul(x, &*self.v_proj)?
            .to_dtype(self.dtype)?;

        // Reshape for multi-head attention
        let q = reshape_for_attn(q, b_sz, seq_len, self.n_head, self.head_dim)?;
        let k = reshape_for_attn(k, b_sz, seq_len, self.n_kv_head, self.head_dim)?;
        let v = reshape_for_attn(v, b_sz, seq_len, self.n_kv_head, self.head_dim)?;

        // RoPE
        let (q, k) = self.rotary.forward(&q, &k, position_offsets)?;

        // SDPA with cache
        let y = sdpa_with_cache(
            &q,
            &k,
            &v,
            mask,
            kv_cache,
            self.paged_attn.as_ref(),
            metadata,
            &self.sdpa_params,
        )?;

        // Reshape output and project
        let y = reshape_attn_output(y, b_sz, seq_len, mask.is_some())?;
        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.o_proj)
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

struct DecoderLayer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    moe: SparseMoeBlock,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        // Pre-norm attention
        let residual = &x;
        let x = self.attn_norm.forward(&x)?;
        let x = self.attn.forward(&x, mask, position_offsets, kv_cache, metadata)?;
        let x = (x + residual)?;

        // Pre-norm MoE
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.moe.forward(&x)?;
        x + residual
    }
}

// =============================================================================
// Mixtral Model
// =============================================================================

/// Mixtral model weights (Mixture of Experts).
///
/// Stateless - cache is passed in, not owned.
pub struct Mixtral {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Option<RmsNorm>,
    output: Option<Arc<dyn QuantMethod>>,
    device: Device,
    max_seq_len: usize,
    num_layers: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

// =============================================================================
// Model Trait Implementations
// =============================================================================

impl Model for Mixtral {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TransformerModel for Mixtral {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(tokens)
    }

    fn transform(
        &self,
        mut hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let position_offsets: Vec<usize> = vec![ctx.position_offset];

        // Compute causal mask
        let mask = CausalMasker.make_causal_mask_as(
            ctx.seq_len,
            hidden.device(),
            &position_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Run through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let layer_metadata = ctx
                .paged_attn
                .as_ref()
                .map(|pa| (pa.kv_cache[i].clone(), pa.metadata));

            hidden = layer.forward(
                hidden,
                mask.as_ref(),
                &position_offsets,
                &mut cache[i],
                layer_metadata,
            )?;
        }

        Ok(hidden)
    }
}

impl LanguageModel for Mixtral {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        match (&self.norm, &self.output) {
            (Some(norm), Some(output)) => {
                let x = norm.forward(&hidden)?;
                MatMul.qmethod_matmul(&x.contiguous()?, &**output)
            }
            _ => {
                // Pipeline parallelism: non-last stage returns hidden states
                Ok(hidden)
            }
        }
    }
}

// =============================================================================
// GGUF Loading
// =============================================================================

/// Mixtral GGUF metadata.
struct MixtralProps {
    n_expert: usize,
    n_expert_used: usize,
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rope_dim: usize,
    rms_norm_eps: f64,
    max_seq_len: usize,
    rope_freq_base: f32,
    head_dim: usize,
}

impl MixtralProps {
    fn from_metadata(metadata: &ContentMetadata) -> std::result::Result<Self, anyhow::Error> {
        metadata.verify_arch("llama")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
        ];
        metadata.has_required_keys(&required)?;

        let embed_len = metadata.get_value::<u32>("embedding_length")? as usize;
        let head_count = metadata.get_value::<u32>("attention.head_count")? as usize;

        let head_dim = metadata
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        let n_expert = metadata.get_value::<u32>("expert_count").ok().unwrap_or(0) as usize;
        let n_expert_used = metadata
            .get_value::<u32>("expert_used_count")
            .ok()
            .unwrap_or(0) as usize;

        if n_expert <= 1 {
            anyhow::bail!(
                "Mixtral requires expert_count > 1, got {}. Use Llama for dense models.",
                n_expert
            );
        }

        Ok(Self {
            n_expert,
            n_expert_used,
            head_count,
            head_count_kv: metadata.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: metadata.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rope_dim: metadata.get_value::<u32>("rope.dimension_count")? as usize,
            rms_norm_eps: metadata.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            max_seq_len: metadata
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(4096) as usize,
            rope_freq_base: metadata
                .get_value::<f32>("rope.freq_base")
                .ok()
                .unwrap_or(10_000.0),
            head_dim,
        })
    }
}

impl ModelConfig::FromGGUF for Mixtral {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        // Parse metadata
        let metadata = ContentMetadata {
            path_prefix: "llama",
            metadata: ct.get_metadata(),
        };
        let props = MixtralProps::from_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Determine layer range for pipeline parallelism
        let layer_range = layer_range.unwrap_or(0..props.block_count);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(props.block_count);

        if layer_start > 0 || layer_end < props.block_count {
            tracing::info!(
                "Pipeline parallelism: loading Mixtral layers {}..{} of {} total",
                layer_start,
                layer_end,
                props.block_count
            );
        }

        // Load embeddings
        let tok_embeddings = {
            let weight = ct.tensor("token_embd.weight", device)?;
            Embedding::new(weight.dequantize(device)?, props.embedding_length)
        };

        // PP: Only load norm and output for last stage
        let is_last_stage = layer_end >= props.block_count;
        let norm = if is_last_stage {
            Some(RmsNorm::from_qtensor(
                ct.tensor("output_norm.weight", device)?,
                props.rms_norm_eps as f32,
            )?)
        } else {
            None
        };
        let output = if is_last_stage {
            Some(if ct.has_tensor("output.weight") {
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ct.tensor("output.weight", device)?),
                    b: None,
                })?) as Arc<dyn QuantMethod>
            } else {
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(ct.tensor("token_embd.weight", device)?),
                    b: None,
                })?) as Arc<dyn QuantMethod>
            })
        } else {
            None
        };

        // Create RoPE embeddings per device
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                e.insert(Arc::new(RotaryEmbedding::new(
                    props.rope_freq_base,
                    props.rope_dim,
                    props.max_seq_len,
                    layer_device,
                    false,
                    dtype,
                )?));
            }
        }

        // Load decoder layers
        let mut layers = Vec::with_capacity(layer_end - layer_start);

        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading Mixtral layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&layer_device.location())
                .expect("No RoPE for device")
                .clone();

            // Load attention weights
            let q_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_q.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            let k_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_k.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            let v_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_v.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            let o_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(
                    ct.tensor(&format!("{prefix}.attn_output.weight"), layer_device)?,
                ),
                b: None,
            })?) as Arc<dyn QuantMethod>;

            // Load MoE gate
            let gate = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(
                    ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), layer_device)?,
                ),
                b: None,
            })?) as Arc<dyn QuantMethod>;

            // Load experts
            let experts = load_experts(&mut ct, &prefix, &props, layer_device)?;

            // Load norms
            let attn_norm = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), layer_device)?,
                props.rms_norm_eps as f32,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), layer_device)?,
                props.rms_norm_eps as f32,
            )?;

            // Paged attention
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(props.head_dim, layer_device, None)?)
                }
            };

            layers.push(DecoderLayer {
                attn_norm,
                attn: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    n_head: props.head_count,
                    n_kv_head: props.head_count_kv,
                    head_dim: props.head_dim,
                    rotary,
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: props.head_count / props.head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (props.head_dim as f32).sqrt(),
                        sliding_window: None,
                    },
                    dtype,
                },
                ffn_norm,
                moe: SparseMoeBlock {
                    gate,
                    experts,
                    n_expert_used: props.n_expert_used,
                },
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: props.max_seq_len,
            num_layers: props.block_count,
            mapper: Some(mapper),
            dtype,
        })
    }
}

/// Load expert MLPs from GGUF.
///
/// Supports both stacked experts (ffn_gate_exps) and individual experts (ffn_gate.{i}).
fn load_experts<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    props: &MixtralProps,
    device: &Device,
) -> Result<Vec<Mlp>> {
    use candle_core::quantized::QTensor;

    let mut experts = Vec::with_capacity(props.n_expert);

    // Try stacked experts first, fall back to individual
    match ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device) {
        Ok(gate_exps) => {
            let down_exps = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
            let up_exps = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;

            let gate_chunks = gate_exps.dequantize(device)?.chunk(props.n_expert, 0)?;
            let down_chunks = down_exps.dequantize(device)?.chunk(props.n_expert, 0)?;
            let up_chunks = up_exps.dequantize(device)?.chunk(props.n_expert, 0)?;

            let gate_type = gate_exps.dtype();
            let down_type = down_exps.dtype();
            let up_type = up_exps.dtype();

            for ((gate, down), up) in gate_chunks
                .into_iter()
                .zip(down_chunks)
                .zip(up_chunks)
            {
                experts.push(Mlp::from_weights(
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(QTensor::quantize(&gate, gate_type)?),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(QTensor::quantize(&up, up_type)?),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(QTensor::quantize(&down, down_type)?),
                        b: None,
                    })?),
                    Activation::Silu,
                ));
            }
        }
        Err(_) => {
            // Individual expert weights
            for i in 0..props.n_expert {
                let gate = ct.tensor(&format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                let down = ct.tensor(&format!("{prefix}.ffn_down.{i}.weight"), device)?;
                let up = ct.tensor(&format!("{prefix}.ffn_up.{i}.weight"), device)?;

                experts.push(Mlp::from_weights(
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(gate),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(up),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(down),
                        b: None,
                    })?),
                    Activation::Silu,
                ));
            }
        }
    }

    Ok(experts)
}
