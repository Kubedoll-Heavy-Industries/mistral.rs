#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSeek V2 model with unified loading from GGUF and safetensors formats.
//!
//! DeepSeek V2 features:
//! - **Multi-Latent Attention (MLA)**: Compresses KV cache via latent vectors
//! - **GroupLimitedGreedy MoE routing**: Experts grouped with per-group limits
//! - Pre-norm with RmsNorm
//! - RoPE with YaRN scaling
//!
//! # Multi-Latent Attention (MLA)
//!
//! MLA compresses the KV cache by projecting to a smaller latent space:
//! ```text
//! Standard: K, V are [batch, seq, num_kv_heads, head_dim]
//! MLA:      KV_compressed is [batch, seq, kv_lora_rank]
//!           K, V are reconstructed on-the-fly for attention
//! ```
//!
//! This significantly reduces memory for long sequences while maintaining
//! model quality through learned up-projection.
//!
//! # Loading Methods
//!
//! - **Safetensors**: Uses `DeepSeekV2::from_safetensors()`
//! - **GGUF**: Not yet implemented (TODO)

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{
        embedding, reshape_attn_output, Activation, CausalMasker, DeepSeekV2RopeConfig,
        DeepSeekV2RopeScaling, DeepSeekV2RotaryEmbedding, FeedForward, MatMul, Mlp, RmsNorm, Sdpa,
    },
    layers_masker::PastKvLenCache,
    models::{LanguageModel, Model as ModelTrait, TransformContext, TokenizerModel},
    moe::{GroupLimitedMoE, MoE, MoELayerConfig, SoftmaxTopK},
    ops::SplitOp,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

// =============================================================================
// Configuration
// =============================================================================

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

/// DeepSeek V2 configuration (for safetensors loading).
#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV2Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) n_shared_experts: Option<usize>,
    pub(crate) n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub(crate) attention_bias: bool,
    pub(crate) q_lora_rank: Option<usize>,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) n_group: usize,
    pub(crate) topk_group: usize,
}

impl DeepSeekV2Config {
    /// Total dimension of Q head (non-positional + positional parts).
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    /// Compute softmax scale accounting for YaRN scaling.
    fn softmax_scale(&self) -> f32 {
        let mut softmax_scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(DeepSeekV2RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }
        softmax_scale
    }

    /// Create MoE layer config from model config.
    fn to_moe_config(&self) -> MoELayerConfig {
        MoELayerConfig {
            hidden_size: self.hidden_size,
            num_experts: self.n_routed_experts.unwrap_or(0),
            num_experts_per_tok: self.num_experts_per_tok.unwrap_or(1),
            moe_intermediate_size: self.moe_intermediate_size,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
        }
    }
}

// =============================================================================
// Multi-Latent Attention (MLA)
// =============================================================================

/// Query projection that may use LoRA compression.
enum QProj {
    /// Direct projection from hidden_size to q_head_dim * num_heads.
    Plain(Arc<dyn QuantMethod>),
    /// LoRA-style compression: hidden -> q_lora_rank -> norm -> q_head_dim * num_heads.
    Lora {
        a: Arc<dyn QuantMethod>,
        norm: RmsNorm,
        b: Arc<dyn QuantMethod>,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => {
                b.forward_autocast(&norm.forward(&a.forward_autocast(xs)?)?)
            }
            Self::Plain(lin) => lin.forward_autocast(xs),
        }
    }
}

/// Multi-Latent Attention (MLA) module.
///
/// MLA compresses KV cache by projecting to a smaller latent space:
/// - Input is projected to compressed KV latent (`kv_a_proj_with_mqa`)
/// - The latent is split into `kv_compressed` and `k_pe` (positional encoding part)
/// - `kv_compressed` is normalized and up-projected to `k_nope` and `v`
/// - Final K is `[k_nope, k_pe]` (concatenated)
///
/// This significantly reduces KV cache memory while maintaining quality.
struct MlaAttention {
    /// Query projection (may be LoRA-compressed).
    q: QProj,
    /// Down projection to KV latent space (+ MQA for k_pe).
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    /// LayerNorm for KV latent before up-projection.
    kv_a_layernorm: RmsNorm,
    /// Up projection from KV latent to K_nope and V.
    kv_b_proj: Arc<dyn QuantMethod>,
    /// Output projection.
    o_proj: Arc<dyn QuantMethod>,
    /// Rotary embeddings for positional encoding.
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    /// Configuration reference for dimensions.
    cfg: DeepSeekV2Config,
    /// Total Q head dimension (qk_nope_head_dim + qk_rope_head_dim).
    q_head_dim: usize,
    /// Optional paged attention module.
    paged_attn: Option<PagedAttention>,
    /// SDPA parameters.
    sdpa_params: SdpaParams,
    /// Number of attention heads (after tensor parallelism).
    num_attention_heads: usize,
}

impl MlaAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();

        // Query projection (plain or LoRA)
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = ReplicatedLayer::new(
                    cfg.hidden_size,
                    lora_rank,
                    &cfg.quantization_config,
                    cfg.attention_bias,
                    mapper.set_device(layer_idx, vb.pp("q_a_proj"), loading_isq),
                )?;
                let norm = RmsNorm::new(
                    lora_rank,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp("q_a_layernorm"), false),
                )?;
                let b = ColumnParallelLayer::new(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    mapper.set_device(layer_idx, vb.pp("q_b_proj"), loading_isq),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            )?),
        };

        // KV latent projection (output: kv_lora_rank + qk_rope_head_dim)
        let kv_a_proj_with_mqa = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp("kv_a_proj_with_mqa"), loading_isq),
        )?;

        // LayerNorm for KV latent
        let kv_a_layernorm = RmsNorm::new(
            cfg.kv_lora_rank,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("kv_a_layernorm"), false),
        )?;

        // KV up-projection (output: num_heads * (qk_nope_head_dim + v_head_dim))
        let kv_b_proj = ColumnParallelLayer::new(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("kv_b_proj"), loading_isq),
        )?;

        // Output projection
        let o_proj = RowParallelLayer::new(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            paged_attn,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            sdpa_params: SdpaParams {
                n_kv_groups: 1, // MLA has same K/V for all heads
                softcap: None,
                softmax_scale: cfg.softmax_scale(),
                sliding_window: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        // 1. Compute Q and reshape to [bs, num_heads, seq_len, q_head_dim]
        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;

        // 2. Split Q into non-positional and positional parts
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        // 3. Compute compressed KV and split into latent + k_pe
        let mut compressed_kv = self.kv_a_proj_with_mqa.forward_autocast(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;

        // 4. Up-project compressed KV to K_nope and V
        let mut kv = self
            .kv_b_proj
            .forward_autocast(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
        kv = kv
            .reshape((
                bs,
                seq_len,
                self.num_attention_heads,
                self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
            ))?
            .transpose(1, 2)?;

        // 5. Split into K_nope and V
        let kv_split = kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let mut v = kv_split[1].clone();

        // 6. Apply RoPE to positional parts only
        (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offsets)?;

        // 7. Concatenate Q = [q_nope, q_pe] and K = [k_nope, k_pe (expanded)]
        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
        let mut k = Tensor::cat(
            &[&k_nope, &k_pe.repeat((1, self.num_attention_heads, 1, 1))?],
            D::Minus1,
        )?
        .contiguous()?;

        // 8. Compute attention
        let mut attn_out = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => {
                    // Pad V to match Q head_dim for paged attention
                    let v = v
                        .pad_with_zeros(D::Minus1, 0, self.q_head_dim - self.cfg.v_head_dim)?
                        .contiguous()?;
                    paged_attn
                        .forward(
                            &q,
                            &k,
                            &v,
                            attention_mask,
                            Some(key_cache),
                            Some(value_cache),
                            input_metadata,
                            &self.sdpa_params,
                            None,
                        )?
                        .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                }
                None => {
                    // No metadata - likely generating imatrix
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    let v = v
                        .pad_with_zeros(D::Minus1, 0, self.q_head_dim - self.cfg.v_head_dim)?
                        .contiguous()?;
                    paged_attn
                        .forward(
                            &q,
                            &k,
                            &v,
                            attention_mask,
                            None,
                            None,
                            &input_metadata,
                            &self.sdpa_params,
                            None,
                        )?
                        .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                }
            },
            None => {
                // Eager attention with KV cache
                (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, attention_mask, None, &self.sdpa_params)?
            }
        };

        // 9. Reshape output
        attn_out = reshape_attn_output(attn_out, bs, seq_len, attention_mask.is_some())?;

        // 10. Output projection
        self.o_proj.forward_autocast(&attn_out)
    }
}

// =============================================================================
// MoE FFN Layer
// =============================================================================

/// MoE with standard softmax top-k routing.
type DeepSeekSoftmaxMoE = MoE<SoftmaxTopK>;

/// FFN layer that can be either MoE or dense MLP.
enum DeepSeekFfn {
    /// MoE layer with softmax top-k routing (Greedy method).
    SoftmaxMoE(DeepSeekSoftmaxMoE),
    /// MoE layer with group-limited greedy routing.
    GroupLimitedMoE(GroupLimitedMoE),
    /// Standard dense MLP layer.
    Mlp(Mlp),
}

impl FeedForward for DeepSeekFfn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::SoftmaxMoE(moe) => moe.forward(xs),
            Self::GroupLimitedMoE(moe) => moe.forward(xs),
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

impl DeepSeekFfn {
    /// Get mutable references to ISQ layers for quantization.
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            Self::SoftmaxMoE(moe) => moe.get_isq_layers(),
            Self::GroupLimitedMoE(moe) => moe.get_isq_layers(),
            Self::Mlp(mlp) => vec![&mut mlp.gate, &mut mlp.up, &mut mlp.down],
        }
    }

    /// Get a reference to the gate for residual tensors.
    fn gate(&self) -> Option<&Arc<dyn QuantMethod>> {
        match self {
            Self::SoftmaxMoE(moe) => Some(moe.gate()),
            Self::GroupLimitedMoE(moe) => Some(moe.gate()),
            Self::Mlp(_) => None,
        }
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

struct DeepSeekDecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: MlaAttention,
    ffn: DeepSeekFfn,
}

impl DeepSeekDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = MlaAttention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;

        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;

        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;

        // Build FFN (MoE or MLP)
        let ffn = if cfg.n_routed_experts.is_some()
            && layer_idx >= cfg.first_k_dense_replace
            && layer_idx % cfg.moe_layer_freq == 0
        {
            let vb_mlp = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or_else(|| vb.device().clone());
            let moe_config = cfg.to_moe_config();

            // Build shared expert if configured
            let shared_expert = if let Some(n_shared_experts) = cfg.n_shared_experts {
                let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
                Some(Mlp::new(
                    vb_mlp.pp("shared_experts"),
                    cfg.hidden_size,
                    intermediate_size,
                    &cfg.quantization_config,
                    cfg.hidden_act,
                    comm,
                )?)
            } else {
                None
            };

            // Create MoE layer based on routing method
            match cfg.topk_method {
                TopkMethod::Greedy => {
                    let mut moe = DeepSeekSoftmaxMoE::new(
                        &moe_config,
                        vb_mlp,
                        layer_device,
                        comm,
                        loading_isq,
                    )?;
                    if let Some(shared) = shared_expert {
                        moe = moe.with_shared_expert(shared);
                    }
                    DeepSeekFfn::SoftmaxMoE(moe)
                }
                TopkMethod::GroupLimitedGreedy => {
                    let mut moe = GroupLimitedMoE::new(
                        &moe_config,
                        vb_mlp,
                        layer_device,
                        comm,
                        loading_isq,
                        cfg.n_group,
                        cfg.topk_group,
                    )?;
                    if let Some(shared) = shared_expert {
                        moe = moe.with_shared_expert(shared);
                    }
                    DeepSeekFfn::GroupLimitedMoE(moe)
                }
            }
        } else {
            DeepSeekFfn::Mlp(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            ffn,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        // Pre-norm -> Attention -> Residual
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;

        // Pre-norm -> FFN -> Residual
        let residual = &xs;
        let xs = self.ffn.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

// =============================================================================
// DeepSeek V2 Model
// =============================================================================

/// DeepSeek V2 model with Multi-Latent Attention and MoE.
pub struct DeepSeekV2 {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: Option<RmsNorm>,
    lm_head: Option<Arc<dyn QuantMethod>>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    num_layers: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    dtype: DType,
}

// =============================================================================
// Model Trait Implementations
// =============================================================================

impl ModelTrait for DeepSeekV2 {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TokenizerModel<[KvCache]> for DeepSeekV2 {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn kv_dim(&self) -> usize {
        self.cfg.k_head_dim * self.cfg.num_kv_heads
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(tokens)
    }

    fn transform(
        &self,
        mut hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let position_offsets: Vec<usize> = vec![ctx.position_offset];

        // Compute causal mask
        let mask = CausalMasker.make_causal_mask_matrix(
            &Tensor::zeros((1, hidden.dims()[1]), DType::U32, hidden.device())?,
            &position_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
            self.cfg.num_attn_heads,
        )?;
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Empty flash params (DeepSeek doesn't use flash attention in this impl)
        let empty_flash = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: Default::default(),
            cumulative_seqlens_k: Default::default(),
            causal: false,
        };
        let flash_params = ctx.flash_params.unwrap_or(&empty_flash);

        // Run through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = self.mapper.map(hidden, i)?;

            let layer_metadata = ctx
                .paged_attn
                .as_ref()
                .map(|pa| (pa.kv_cache[i].clone(), pa.metadata));

            hidden = layer.forward(
                &hidden,
                mask.as_ref()
                    .map(|m| m.to_device(hidden.device()).unwrap())
                    .as_ref(),
                &position_offsets,
                &mut cache[i],
                layer_metadata,
                flash_params,
            )?;
        }

        Ok(hidden)
    }
}

impl LanguageModel<[KvCache]> for DeepSeekV2 {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let x = norm.forward(&hidden)?;
                MatMul.qmethod_matmul(&x.contiguous()?, &**lm_head)
            }
            _ => {
                // Pipeline parallelism: non-last stage returns hidden states
                Ok(hidden)
            }
        }
    }
}

// =============================================================================
// ISQ Support
// =============================================================================

impl IsqModel for DeepSeekV2 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        if let Some(ref mut lm_head) = self.lm_head {
            tensors.push((lm_head, None));
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections
            match &mut layer.attn.q {
                QProj::Plain(q) => {
                    tensors.push((q, Some(i)));
                }
                QProj::Lora { a, norm: _, b } => {
                    tensors.push((a, Some(i)));
                    tensors.push((b, Some(i)));
                }
            }
            tensors.push((&mut layer.attn.kv_a_proj_with_mqa, Some(i)));
            tensors.push((&mut layer.attn.kv_b_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));

            // FFN layers
            for layer_ref in layer.ffn.get_isq_layers() {
                tensors.push((layer_ref, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        if let Some(ref mut lm_head) = self.lm_head {
            tensors.push((lm_head, None));
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for layer_ref in layer.ffn.get_isq_layers() {
                tensors.push((layer_ref, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        if let Some(ref norm) = self.norm {
            uvb_m.pp("norm").add(norm);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_layernorm);

            // Add gate tensor for MoE layers
            if let Some(gate) = layer.ffn.gate() {
                uvb_l.pp("mlp").pp("gate").add(gate);
            }

            // Add Q LayerNorm for LoRA variant
            match &layer.attn.q {
                QProj::Plain(_) => (),
                QProj::Lora { a: _, norm, b: _ } => {
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                }
            }
        }

        uvb.to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        if let Some(ref norm) = self.norm {
            uvb_m.pp("norm").add(norm);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_layernorm);

            // Add gate tensor for MoE layers
            if let Some(gate) = layer.ffn.gate() {
                uvb_l.pp("mlp").pp("gate").add(gate);
            }

            // Add all attention projections
            match &layer.attn.q {
                QProj::Plain(q) => {
                    uvb_l.pp("self_attn").pp("q_proj").add(q);
                }
                QProj::Lora { a, norm, b } => {
                    uvb_l.pp("self_attn").pp("q_a_proj").add(a);
                    uvb_l.pp("self_attn").pp("q_a_layernorm").add(norm);
                    uvb_l.pp("self_attn").pp("q_b_proj").add(b);
                }
            }
            uvb_l
                .pp("self_attn")
                .pp("kv_a_proj_with_mqa")
                .add(&layer.attn.kv_a_proj_with_mqa);
            uvb_l
                .pp("self_attn")
                .pp("kv_b_proj")
                .add(&layer.attn.kv_b_proj);
            uvb_l.pp("self_attn").pp("o_proj").add(&layer.attn.o_proj);
        }

        Some(uvb.to_safetensors())
    }
}

// =============================================================================
// NormalModel Support (for NormalPipeline compatibility)
// =============================================================================

impl NormalModel for DeepSeekV2 {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;

        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?;
        }

        let xs = xs.to_device(&self.device)?;
        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let xs = norm.forward(&xs)?;
                extract_logits(&lm_head.forward_autocast(&xs)?, context_lens)
            }
            _ => Ok(xs),
        }
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::lora::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for DeepSeekV2 {}

// =============================================================================
// Safetensors Loading
// =============================================================================

impl DeepSeekV2 {
    /// Create a new DeepSeek V2 model from safetensors weights.
    pub fn from_safetensors(
        cfg: &DeepSeekV2Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb.dtype();

        // Embeddings
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        // LM head
        let lm_head = if !cfg.tie_word_embeddings {
            Some(ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?)
        } else {
            Some(ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?)
        };

        // Output norm
        let norm = Some(RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?);

        // Create RoPE embeddings per device
        let mut ropes = HashMap::new();
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(&rope_cfg, dtype, device)?),
            );
        }

        // Load decoder layers
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DeepSeekDecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(cfg.v_head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DeepSeekDecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })?;

        // Pre-compute world_size before moving mapper
        let world_size = mapper.get_comm_for(0)?.world_size();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device.clone(),
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_attention_heads / world_size).max(1),
                num_attn_heads: (cfg.num_attention_heads / world_size).max(1),
                sliding_window: None,
                k_head_dim: cfg.q_head_dim(),
                v_head_dim: if matches!(
                    attention_mechanism,
                    AttentionImplementation::PagedAttention
                ) {
                    cfg.q_head_dim()
                } else {
                    cfg.v_head_dim
                },
            },
            dtype,
        })
    }

    /// Legacy constructor for compatibility.
    pub fn new(
        cfg: &DeepSeekV2Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::from_safetensors(cfg, vb, is_gptx, normal_loading_metadata, attention_mechanism)
    }
}

// =============================================================================
// Deprecated: Old Model type (use DeepSeekV2 instead)
// =============================================================================

#[deprecated(since = "0.8.0", note = "Use `DeepSeekV2::from_safetensors()` instead")]
pub type Model = DeepSeekV2;
