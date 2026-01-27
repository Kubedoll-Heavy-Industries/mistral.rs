#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Mixtral model with unified loading from GGUF and safetensors formats.
//!
//! Mixtral uses the Llama architecture with MoE (Mixture of Experts) replacing
//! the standard MLP. Each token is routed to top-k experts, combining their outputs.
//!
//! This is a fundamentally different compute pattern from dense models:
//! - Sparse activation: only k of n experts run per token
//! - Different memory access patterns
//! - Different parallelism strategies
//!
//! # Loading Methods
//!
//! - **GGUF**: Uses `Mixtral::from_gguf()` with loop-based expert execution
//! - **Safetensors**: Uses `Mixtral::from_safetensors()` with optimized MoE kernels
//!
//! Both paths use the unified `SoftmaxTopK` routing strategy.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, GgufMatMul, QuantMethod, QuantMethodConfig, QuantizedConfig,
    ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    gguf::Content,
    layers::{
        embedding, reshape_attn_output, reshape_for_attn, sdpa_with_cache, Activation,
        CausalMasker, FeedForward, MatMul, RmsNorm, RotaryEmbedding,
    },
    layers_masker::PastKvLenCache,
    models::{LanguageModel, Model as ModelTrait, TransformContext, TransformerModel},
    moe::{
        routing::{RoutingConfig, SoftmaxTopK},
        LoadedExpertWeights, MoE, MoEExperts, MoELayerConfig, QuantProperties,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::gguf_metadata::ContentMetadata,
    utils::model_config as ModelConfig,
    utils::progress::{new_multi_progress, NiceProgressBar},
    utils::unvarbuilder::UnVarBuilder,
};

serde_default_fn!(bool, word_emb_default, false);

// =============================================================================
// Configuration
// =============================================================================

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

impl Config {
    /// Head dimension for attention.
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Create MoE layer config from model config.
    fn to_moe_config(&self) -> MoELayerConfig {
        MoELayerConfig {
            hidden_size: self.hidden_size,
            num_experts: self.num_local_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.intermediate_size,
            norm_topk_prob: true, // Mixtral normalizes top-k weights
            routed_scaling_factor: 1.0,
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
        }
    }
}

// =============================================================================
// Unified FFN Layer (MoE for both GGUF and Safetensors)
// =============================================================================

/// Unified FFN layer for Mixtral models.
///
/// Both GGUF and safetensors loading now use the same `MoE<SoftmaxTopK>` type.
/// Backend selection (Fused/Fast/Slow) is determined by tensor properties (device,
/// quantization) rather than file format.
type MixtralFfn = MoE<SoftmaxTopK>;

// =============================================================================
// Unified Attention Layer
// =============================================================================

struct MixtralAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl MixtralAttention {
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
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

        // RoPE
        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        // SDPA with cache
        let mut attn_output = sdpa_with_cache(
            &q,
            &k,
            &v,
            attention_mask,
            kv_cache,
            self.paged_attn.as_ref(),
            metadata,
            &self.sdpa_params,
        )?;

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
// Unified Decoder Layer
// =============================================================================

struct MixtralDecoderLayer {
    self_attn: MixtralAttention,
    moe: MixtralFfn,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MixtralDecoderLayer {
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
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
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.moe.forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

// =============================================================================
// Mixtral Model (Unified)
// =============================================================================

/// Mixtral model weights (Mixture of Experts).
///
/// Supports both GGUF and safetensors loading via `FromGGUF` and `FromSafetensors` traits.
/// Uses unified `SoftmaxTopK` routing for both paths.
pub struct Mixtral {
    embed_tokens: Embedding,
    layers: Vec<MixtralDecoderLayer>,
    norm: Option<RmsNorm>,
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

impl ModelTrait for Mixtral {
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

        // Run through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let layer_metadata = ctx
                .paged_attn
                .as_ref()
                .map(|pa| (pa.kv_cache[i].clone(), pa.metadata));

            // Create empty FlashParams for layers (GGUF models don't use flash attention)
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
                &mut cache[i],
                layer_metadata,
                flash_params,
            )?;
        }

        Ok(hidden)
    }
}

impl LanguageModel for Mixtral {
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

impl IsqModel for Mixtral {
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
            for layer in layer.moe.get_isq_layers() {
                tensors.push((layer, Some(i)));
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
            uvb_l.pp("block_sparse_moe").pp("gate").add(layer.moe.gate());
        }

        uvb.to_safetensors()
    }
}

// =============================================================================
// NormalModel Support (for NormalPipeline compatibility)
// =============================================================================

impl NormalModel for Mixtral {
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
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
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

impl AnyMoeBaseModelMixin for Mixtral {}

// =============================================================================
// GGUF Loading
// =============================================================================

/// Mixtral GGUF metadata.
struct MixtralGgufProps {
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

impl MixtralGgufProps {
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
        let props = MixtralGgufProps::from_metadata(&metadata)
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
        let embed_tokens = {
            let weight = ct.tensor("token_embd.weight", device)?;
            Embedding::new(weight.dequantize(device)?, props.embedding_length)
        };

        // PP: Only load norm and lm_head for last stage
        let is_last_stage = layer_end >= props.block_count;
        let norm = if is_last_stage {
            Some(RmsNorm::from_qtensor(
                ct.tensor("output_norm.weight", device)?,
                props.rms_norm_eps as f32,
            )?)
        } else {
            None
        };
        let lm_head = if is_last_stage {
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

            // Load experts into LoadedExpertWeights
            let loaded_experts = load_gguf_experts(&mut ct, &prefix, &props, layer_device)?;

            // Get comm for tensor parallelism (single-node for GGUF)
            let comm = mapper.get_comm_for(layer_idx)?;

            // Create unified MoEExperts with automatic backend selection
            let moe_experts = MoEExperts::from_loaded(
                loaded_experts,
                props.n_expert_used,
                &comm,
                Activation::Silu,
            )?;

            // Create unified MoE layer
            let routing_config = RoutingConfig::new_normalized(
                props.n_expert,
                props.n_expert_used,
            );
            let moe = MoE::from_parts(gate, moe_experts, routing_config);

            // Load norms
            let input_layernorm = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), layer_device)?,
                props.rms_norm_eps as f32,
            )?;
            let post_attention_layernorm = RmsNorm::from_qtensor(
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

            layers.push(MixtralDecoderLayer {
                self_attn: MixtralAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_heads: props.head_count,
                    num_kv_heads: props.head_count_kv,
                    head_dim: props.head_dim,
                    rotary_emb: rotary,
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: props.head_count / props.head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (props.head_dim as f32).sqrt(),
                        sliding_window: None,
                    },
                    dtype,
                },
                input_layernorm,
                post_attention_layernorm,
                moe,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: None,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(props.block_count, props.max_seq_len)),
            max_seq_len: props.max_seq_len,
            num_layers: props.block_count,
            mapper: Some(mapper),
            cfg: ModelConfigMetadata {
                max_seq_len: props.max_seq_len,
                num_layers: props.block_count,
                hidden_size: props.embedding_length,
                num_kv_heads: props.head_count_kv,
                num_attn_heads: props.head_count,
                sliding_window: None,
                k_head_dim: props.head_dim,
                v_head_dim: props.head_dim,
            },
            dtype,
        })
    }
}

/// Load expert weights from GGUF into LoadedExpertWeights.
///
/// This function loads expert weights and returns them in a format suitable for
/// `MoEExperts::from_loaded()`, enabling unified backend selection.
fn load_gguf_experts<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    props: &MixtralGgufProps,
    device: &Device,
) -> Result<LoadedExpertWeights> {
    use candle_core::quantized::{GgmlDType, QTensor};

    let mut gate_proj = Vec::with_capacity(props.n_expert);
    let mut up_proj = Vec::with_capacity(props.n_expert);
    let mut down_proj = Vec::with_capacity(props.n_expert);
    let mut detected_dtype: Option<GgmlDType> = None;

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
            detected_dtype = Some(gate_type);

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
        }
        Err(_) => {
            // Individual expert weights
            for i in 0..props.n_expert {
                let gate = ct.tensor(&format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                let down = ct.tensor(&format!("{prefix}.ffn_down.{i}.weight"), device)?;
                let up = ct.tensor(&format!("{prefix}.ffn_up.{i}.weight"), device)?;

                if detected_dtype.is_none() {
                    detected_dtype = Some(gate.dtype());
                }

                gate_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(gate),
                    b: None,
                })?) as Arc<dyn QuantMethod>);
                up_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(up),
                    b: None,
                })?) as Arc<dyn QuantMethod>);
                down_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(down),
                    b: None,
                })?) as Arc<dyn QuantMethod>);
            }
        }
    }

    // Create quant properties for backend selection
    let quant_properties = QuantProperties::gguf(detected_dtype.unwrap_or(GgmlDType::Q4K));

    Ok(LoadedExpertWeights {
        gate_proj,
        up_proj,
        down_proj,
        quant_properties,
    })
}

// =============================================================================
// Safetensors Loading
// =============================================================================

impl Mixtral {
    /// Create a new Mixtral model from safetensors weights.
    pub fn from_safetensors(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::from_safetensors_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    fn from_safetensors_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    dtype,
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<MixtralDecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .run(
            !(normal_loading_metadata.real_device.is_metal() && cfg.quantization_config.is_none()),
            |layer_idx| {
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
                        PagedAttention::new(head_dim, device, None)
                            .expect("PagedAttention creation failed"),
                    ),
                };
                let comm = mapper
                    .get_comm_for(layer_idx)
                    .expect("Failed to get comm for layer");

                load_safetensors_layer(
                    cfg,
                    &vb_l.pp(layer_idx),
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    paged_attn,
                    rotary_emb,
                    normal_loading_metadata.real_device.clone(),
                    &comm,
                    dtype,
                )
            },
        )?;

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };

        Ok(Self {
            embed_tokens: Embedding::new(embed_tokens.embeddings().clone(), cfg.hidden_size),
            layers,
            norm: Some(norm),
            lm_head: Some(lm_head),
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(cfg.num_hidden_layers, cfg.max_position_embeddings)),
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            mapper: Some(mapper),
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
            },
            dtype,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn load_safetensors_layer(
    cfg: &Config,
    vb: &ShardedVarBuilder,
    mapper: &(dyn DeviceMapper + Send + Sync),
    layer_idx: usize,
    loading_isq: bool,
    paged_attn: Option<PagedAttention>,
    rotary_emb: Arc<RotaryEmbedding>,
    real_device: Device,
    comm: &Arc<mistralrs_quant::Comm>,
    dtype: DType,
) -> Result<MixtralDecoderLayer> {
    let hidden_sz = cfg.hidden_size;
    let num_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim();

    let vb_attn = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
    let q_proj = ColumnParallelLayer::new(
        hidden_sz,
        num_heads * head_dim,
        &cfg.quantization_config,
        false,
        comm,
        vb_attn.pp("q_proj"),
    )?;
    let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm);
    let k_proj = ColumnParallelLayer::new_with_shard(
        hidden_sz,
        num_kv_heads * head_dim,
        &cfg.quantization_config,
        false,
        comm,
        kv_shard,
        vb_attn.pp("k_proj"),
    )?;
    let v_proj = ColumnParallelLayer::new_with_shard(
        hidden_sz,
        num_kv_heads * head_dim,
        &cfg.quantization_config,
        false,
        comm,
        kv_shard,
        vb_attn.pp("v_proj"),
    )?;
    let o_proj = RowParallelLayer::new(
        num_heads * head_dim,
        hidden_sz,
        &cfg.quantization_config,
        false,
        comm,
        vb_attn.pp("o_proj"),
    )?;

    let layer_device = mapper
        .device_for(layer_idx, false)
        .cloned()
        .unwrap_or(real_device);

    // Use the type-safe MoE layer with SoftmaxTopK routing
    let moe = MoE::new(
        &cfg.to_moe_config(),
        mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
        layer_device,
        comm,
        loading_isq,
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

    Ok(MixtralDecoderLayer {
        self_attn: MixtralAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
            },
            dtype,
        },
        input_layernorm,
        post_attention_layernorm,
        moe,
    })
}

// =============================================================================
// Deprecated: Old Model type (use Mixtral instead)
// =============================================================================

#[deprecated(since = "0.8.0", note = "Use `Mixtral::from_safetensors()` instead")]
pub type Model = Mixtral;
