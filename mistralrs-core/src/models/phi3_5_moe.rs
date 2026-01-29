#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Phi-3.5 MoE model with unified loading from safetensors format.
//!
//! Phi-3.5 MoE combines the Phi-3 architecture with Mixture of Experts:
//! - Pre-norm with LayerNorm (not RmsNorm like most MoE models)
//! - Separate Q, K, V projections (not fused QKV like standard Phi-3)
//! - Sliding window attention
//! - **SparseMixer routing** - threshold-based selection with jitter, NOT softmax top-k
//!
//! # Architecture
//!
//! Each decoder layer consists of:
//! 1. Input LayerNorm + Attention (with sliding window) + Residual
//! 2. Post-attention LayerNorm + SparseMixer MoE FFN + Residual
//!
//! The MoE routing uses SparseMixer which:
//! - Always selects exactly 2 experts per token
//! - Uses argmax + threshold masking instead of softmax top-k
//! - Applies jitter noise for regularization
//!
//! # Loading Methods
//!
//! - **Safetensors**: Uses `Phi35Moe::from_safetensors()` with SparseMixerMoE
//!
//! Both paths use the unified `SparseMixer` routing strategy.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{
        embedding, layer_norm, CausalMasker, FeedForward, MatMul, PhiRopeConfig,
        PhiRopeScalingConfig, PhiRotaryEmbedding,
    },
    layers_masker::PastKvLenCache,
    models::{LanguageModel, Model as ModelTrait, TransformContext, TokenizerModel},
    moe::{MoELayerConfig, SparseMixerMoE},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::progress::NiceProgressBar,
    utils::unvarbuilder::UnVarBuilder,
};

use crate::layers::{reshape_attn_output, reshape_for_attn, Sdpa};

serde_default_fn!(bool, word_emb_default, false);

// =============================================================================
// Configuration
// =============================================================================

/// Phi-3.5 MoE configuration (for safetensors loading).
///
/// This is the JSON config.json structure from HuggingFace Phi-3.5 MoE models.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_act: crate::layers::Activation,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rope_scaling: Option<PhiRopeScalingConfig>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) original_max_position_embeddings: usize,

    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) lm_head_bias: bool,
    pub(crate) attention_bias: bool,
    pub(crate) num_local_experts: usize,
    pub(crate) router_jitter_noise: f64,
    #[serde(default = "word_emb_default")]
    pub(crate) tie_word_embeddings: bool,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
            partial_rotary_factor: None,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Create MoE layer config from model config.
    fn to_moe_config(&self) -> MoELayerConfig {
        MoELayerConfig {
            hidden_size: self.hidden_size,
            num_experts: self.num_local_experts,
            num_experts_per_tok: 2, // Phi-3.5 MoE always uses top-2
            moe_intermediate_size: self.intermediate_size,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
        }
    }
}

// =============================================================================
// Attention Layer
// =============================================================================

struct Phi35MoeAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Phi35MoeAttention {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
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

        // Reshape for multi-head attention
        let q = reshape_for_attn(q, b_sz, q_len, self.num_heads, self.head_dim)?;
        let k = reshape_for_attn(k, b_sz, q_len, self.num_kv_heads, self.head_dim)?;
        let v = reshape_for_attn(v, b_sz, q_len, self.num_kv_heads, self.head_dim)?;

        // Apply RoPE
        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids)?;

        // Attention with cache
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
                    None,
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix
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
                        None,
                    )?
                }
            },
            _ => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, attention_mask, None, &self.sdpa_params)?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = reshape_attn_output(attn_output, b_sz, q_len, attention_mask.is_some())?;

        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// =============================================================================
// Decoder Layer
// =============================================================================

struct Phi35MoeDecoderLayer {
    self_attn: Phi35MoeAttention,
    moe: SparseMixerMoE,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl Phi35MoeDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Phi35MoeAttention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;

        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        // Use unified SparseMixerMoE for Phi-3.5 style routing
        let moe = SparseMixerMoE::new(
            &cfg.to_moe_config(),
            mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
            layer_device,
            comm,
            cfg.router_jitter_noise,
        )?;

        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            position_ids,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

// =============================================================================
// Phi-3.5 MoE Model
// =============================================================================

/// Phi-3.5 MoE model weights.
///
/// Uses unified `SparseMixer` routing for threshold-based expert selection.
pub struct Phi35Moe {
    embed_tokens: Embedding,
    layers: Vec<Phi35MoeDecoderLayer>,
    norm: Option<LayerNorm>,
    lm_head: Option<Arc<dyn QuantMethod>>,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    num_layers: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    cfg: ModelConfigMetadata,
    dtype: DType,
}

// =============================================================================
// Model Trait Implementations
// =============================================================================

impl ModelTrait for Phi35Moe {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TokenizerModel<[KvCache]> for Phi35Moe {
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

        // Create sliding window causal mask
        let mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            &Tensor::zeros((1, hidden.dims()[1]), DType::U32, hidden.device())?,
            &position_offsets.as_slice() as &dyn PastKvLenCache,
            self.sliding_window,
            self.dtype,
            self.cfg.num_attn_heads,
        )?;
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Position IDs for RoPE
        let position_ids: Vec<usize> = (0..hidden.dims()[1])
            .map(|i| i + ctx.position_offset)
            .collect();

        // Run through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let layer_metadata = ctx
                .paged_attn
                .as_ref()
                .map(|pa| (pa.kv_cache[i].clone(), pa.metadata));

            let empty_flash = FlashParams {
                max_q: 0,
                max_k: 0,
                cumulative_seqlens_q: Default::default(),
                cumulative_seqlens_k: Default::default(),
                causal: false,
            };
            let flash_params = ctx.flash_params.unwrap_or(&empty_flash);
            hidden = layer.forward(
                &hidden,
                mask.as_ref(),
                &position_offsets,
                &position_ids,
                &mut cache[i],
                layer_metadata,
                flash_params,
            )?;
        }

        Ok(hidden)
    }
}

impl LanguageModel<[KvCache]> for Phi35Moe {
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

impl IsqModel for Phi35Moe {
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
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            for moe_layer in layer.moe.get_isq_layers() {
                tensors.push((moe_layer, Some(i)));
            }
        }
        let mapper = self
            .mapper
            .as_ref()
            .expect("Model must have a mapper for get_layers");
        (tensors, &**mapper)
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
            for moe_layer in layer.moe.get_isq_layers() {
                tensors.push((moe_layer, Some(i)));
            }
        }
        let mapper = self
            .mapper
            .as_ref()
            .expect("Model must have a mapper for get_layers");
        (tensors, &**mapper)
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
                .pp("block_sparse_moe")
                .pp("gate")
                .add(layer.moe.gate());
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

            let uvb_attn = uvb_l.pp("self_attn");
            uvb_attn.pp("q_proj").add(&layer.self_attn.q_proj);
            uvb_attn.pp("k_proj").add(&layer.self_attn.k_proj);
            uvb_attn.pp("v_proj").add(&layer.self_attn.v_proj);
            uvb_attn.pp("o_proj").add(&layer.self_attn.o_proj);
            uvb_l
                .pp("block_sparse_moe")
                .pp("gate")
                .add(layer.moe.gate());
        }

        Some(uvb.to_safetensors())
    }
}

// =============================================================================
// NormalModel Support (for NormalPipeline compatibility)
// =============================================================================

impl NormalModel for Phi35Moe {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.sliding_window,
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
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &position_ids,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?
        }

        let xs = xs.to_device(&self.device)?;
        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let mut xs = norm.forward(&xs)?;
                if let Some(t) = lm_head.quantized_act_type() {
                    xs = xs.to_dtype(t)?;
                }
                extract_logits(&MatMul.qmethod_matmul(&xs, &**lm_head)?, context_lens)
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

impl AnyMoeBaseModelMixin for Phi35Moe {}

// =============================================================================
// Safetensors Loading
// =============================================================================

impl Phi35Moe {
    /// Create a new Phi-3.5 MoE model from safetensors weights.
    pub fn from_safetensors(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb.dtype();

        // Pre-compute world_size for ModelConfigMetadata before moving mapper
        let world_size = mapper.get_comm_for(0)?.world_size();
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        // Create RoPE embeddings per device
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(PhiRotaryEmbedding::new(vb.dtype(), cfg.clone(), device)?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<Phi35MoeDecoderLayer> = NiceProgressBar::<_, 'b'>(
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
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim(), device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            Phi35MoeDecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                normal_loading_metadata.real_device.clone(),
                &comm,
            )
        })?;

        let norm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                cfg.lm_head_bias,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            unreachable!()
        };

        Ok(Self {
            embed_tokens: Embedding::new(embed_tokens.embeddings().clone(), cfg.hidden_size),
            layers,
            norm: Some(norm),
            lm_head: Some(lm_head),
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new_sliding(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
                cfg.sliding_window,
            )),
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            mapper: Some(mapper),
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / world_size,
                num_kv_heads: (cfg.num_key_value_heads / world_size).max(1),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
            },
            dtype,
        })
    }

    /// Legacy constructor for NormalModelLoader compatibility.
    #[deprecated(since = "0.8.0", note = "Use `from_safetensors()` instead")]
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::from_safetensors(cfg, vb, is_gptx, normal_loading_metadata, attention_mechanism)
    }
}

// =============================================================================
// Deprecated: Old Model type (use Phi35Moe instead)
// =============================================================================

#[deprecated(since = "0.8.0", note = "Use `Phi35Moe::from_safetensors()` instead")]
pub type Model = Phi35Moe;
