#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GLM-4 (ChatGLM-4) model implementation.
//!
//! GLM-4 has a unique pre-norm architecture with 4 normalization layers per block:
//! - `input_layernorm` - pre-attention norm
//! - `post_self_attn_layernorm` - norm applied to attention output
//! - `post_attention_layernorm` - pre-FFN norm
//! - `post_mlp_layernorm` - norm applied to FFN output
//!
//! This module provides:
//! - `Model` - Legacy `NormalModel` implementation for safetensors loading (backwards compatible)
//! - `ModelWeights` - Universal architecture with `FromGGUF` + `FromSafetensors` traits
//!
//! The loader infrastructure uses `Model::new()` for safetensors loading via the
//! `NormalModelLoader` trait. The `ModelWeights` implementation is available for
//! GGUF loading and future typed pipeline usage.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use crate::amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp};
use crate::attention::{AttentionConfig, CausalAttention, PositionEncoding};
use crate::device_map::DeviceMapper;
use crate::get_delta_from_lora_ab;
use crate::gguf::Content;
use crate::layers::{
    embedding, Activation, CausalMasker, MatMul, Mlp, PartialRotaryEmbedding,
    RmsNorm, RotaryEmbedding, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{
    standard_embed, LanguageModel, LanguageModelExt, Model as ModelTrait, TransformContext,
    TokenizerModel, TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention};
use crate::pipeline::loaders::{
    GgufNaming, GgufWeightSource, TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{
    extract_logits, EitherCache, IsqModel, KvCache, NormalCache, NormalCacheType,
    NormalLoadingMetadata, NormalModel,
};
use crate::serde_default_fn;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use crate::utils::unvarbuilder::UnVarBuilder;

// =============================================================================
// Safetensors Configuration (JSON config.json)
// =============================================================================

serde_default_fn!(bool, tie_word_embeddings_default, false);
serde_default_fn!(usize, max_position_embeddings_default, 32768);

/// Configuration for GLM-4 model loaded from safetensors.
/// Mirrors the config.json structure from HuggingFace.
#[derive(Debug, Clone, serde::Deserialize, Default, serde::Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(default = "max_position_embeddings_default")]
    pub max_position_embeddings: usize,
    pub attention_bias: Option<bool>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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
// GLM-4 Transformer Block
// =============================================================================

/// GLM-4 transformer block with 4 normalization layers.
///
/// Unlike standard transformers (2 norms), GLM-4 uses:
/// 1. `input_layernorm` - pre-attention norm
/// 2. `post_self_attn_layernorm` - norm on attention output
/// 3. `post_attention_layernorm` - pre-FFN norm
/// 4. `post_mlp_layernorm` - norm on FFN output
pub struct Glm4TransformerBlock {
    pub input_layernorm: RmsNorm,
    pub attention: CausalAttention,
    pub post_self_attn_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
    pub post_mlp_layernorm: RmsNorm,
}

impl Glm4TransformerBlock {
    /// Create a new GLM-4 transformer block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_layernorm: RmsNorm,
        attention: CausalAttention,
        post_self_attn_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        post_mlp_layernorm: RmsNorm,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_self_attn_layernorm,
            post_attention_layernorm,
            mlp,
            post_mlp_layernorm,
        }
    }

    /// Forward pass through the GLM-4 transformer block.
    pub fn forward(
        &self,
        hidden: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        // Pre-norm attention
        let residual = &hidden;
        let x = self.input_layernorm.forward(&hidden)?;

        // Move mask to same device as x (for device mapping support)
        let mask = mask.map(|m| m.to_device(x.device()).unwrap());
        let attn_out = self
            .attention
            .forward(&x, mask.as_ref(), cache, position_offsets, metadata)?;

        // GLM-4 specific: norm on attention output before residual
        let attn_out = self.post_self_attn_layernorm.forward(&attn_out)?;
        let x = (residual + attn_out)?;

        // Pre-norm FFN
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&x)?;

        // GLM-4 specific: norm on FFN output before residual
        let ffn_out = self.post_mlp_layernorm.forward(&ffn_out)?;
        ffn_out + residual
    }
}

impl crate::models::TransformerLayer for Glm4TransformerBlock {
    fn forward(
        &self,
        hidden: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        cache: &mut KvCache,
        paged_attn_meta: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        Glm4TransformerBlock::forward(self, hidden, mask, position_offsets, cache, paged_attn_meta)
    }
}

// =============================================================================
// Legacy Model (for backwards compatibility with NormalModelLoader)
// =============================================================================

use std::iter::zip;

/// GLM-4 RoPE implementation with support for partial rotary factor.
struct Glm4RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl Glm4RotaryEmbedding {
    fn new(
        rope_theta: f32,
        partial_rotary_factor: Option<f32>,
        head_dim: usize,
        max_seq_len: usize,
        dev: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut rotary_dim = head_dim;
        if let Some(factor) = partial_rotary_factor {
            rotary_dim = (factor * head_dim as f32) as usize;
        };

        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &[usize]) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let (s, e) = (*seqlen_offset, *seqlen_offset + seq_len);
            let cos = self.cos.i((s..e, ..))?.contiguous()?;
            let sin = self.sin.i((s..e, ..))?.contiguous()?;
            let xs_rot = xs
                .i((b, .., .., ..self.rotary_dim))?
                .unsqueeze(0)?
                .contiguous()?;
            let xs_pass = xs.i((b, .., .., self.rotary_dim..))?.unsqueeze(0)?;
            let xs_rot = candle_nn::rotary_emb::rope_i(&xs_rot, &cos, &sin).unwrap();
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

/// Legacy attention implementation for the NormalModel interface.
struct LegacyAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Glm4RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: crate::attention::SdpaParams,
}

impl LegacyAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<Glm4RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias.unwrap_or(false),
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias.unwrap_or(false),
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            cfg.attention_bias.unwrap_or(false),
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        assert!(cfg.num_attention_heads >= comm.world_size());
        assert!(cfg.num_attention_heads % comm.world_size() == 0);

        assert!(cfg.num_key_value_heads >= comm.world_size());
        assert!(cfg.num_key_value_heads % comm.world_size() == 0);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: crate::attention::SdpaParams {
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
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
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

        (q, k, v) = if q_len != 1 {
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

        q = self.rotary_emb.apply_rotary_emb(&q, seqlen_offsets)?;
        k = self.rotary_emb.apply_rotary_emb(&k, seqlen_offsets)?;

        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

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
                    Some(flash_params),
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
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
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

/// Legacy decoder layer for the NormalModel interface.
struct LegacyDecoderLayer {
    self_attn: LegacyAttention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
}

impl LegacyDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<Glm4RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = LegacyAttention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new_merged(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            2,
            &cfg.quantization_config,
            cfg.hidden_act,
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

        let post_self_attn_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_self_attn_layernorm"), false),
        )?;
        let post_mlp_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_mlp_layernorm"), false),
        )?;

        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        let hidden_states = self.input_layernorm.forward(xs)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let hidden_states = self.post_self_attn_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;
        residual + hidden_states
    }
}

/// Legacy GLM-4 model for NormalModel interface (safetensors loading via loader).
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<LegacyDecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
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
        let vb_m = vb.pp("model");

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
                Arc::new(Glm4RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.partial_rotary_factor,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    if normal_loading_metadata.loading_isq {
                        DType::F32
                    } else {
                        vb_m.dtype()
                    },
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| -> Result<LegacyDecoderLayer> {
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
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            LegacyDecoderLayer::new(
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
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
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
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|_| {
                cfg.sliding_window
                    .map(|window| NormalCacheType::SlidingWindow { window })
                    .unwrap_or(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
            },
            mapper,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
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
        let mut xs = xs.apply(&self.norm)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        extract_logits(&MatMul.qmethod_matmul(&xs, &*self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("post_self_attn_layernorm")
                .add(&layer.post_self_attn_layernorm);
            uvb_l
                .pp("post_mlp_layernorm")
                .add(&layer.post_mlp_layernorm);
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        names.push(None);
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
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

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer) {
                    continue;
                }

                let intermediate_size = self.layers[layer].mlp.get_params()[1];
                let hidden_size = self.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.layers[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::replicate(
                            self.layers[layer].mlp.get_params(),
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            self.layers[layer].mlp.hidden_act(),
                            &self.mapper.get_comm_for(layer)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.layers[layer].mlp.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.layers[layer].mlp.dtype_device();
            self.layers[layer].mlp = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

// =============================================================================
// Model Weights (Universal Architecture)
// =============================================================================

/// GLM-4 model weights using the universal architecture.
///
/// The model uses pre-norm architecture with:
/// - RMS normalization (4 per block instead of standard 2)
/// - Optional partial RoPE positional embeddings
/// - Gated MLP with SiLU activation
pub struct ModelWeights {
    tok_embeddings: candle_nn::Embedding,
    layers: Vec<Glm4TransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
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
        // Log quantization info if present
        if let Some(quant_cfg) = cfg.quantization_config.as_ref() {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }

        let vb_m = vb.pp("model");
        let quant_cfg = cfg.quantization_config.clone();

        // Load embedding weights
        let tok_embeddings = crate::layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &quant_cfg,
        )?;

        // Load output norm
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Load output weights (may be tied to embeddings)
        let output: Arc<dyn QuantMethod> = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                &quant_cfg,
                vb.pp("lm_head"),
            )?
        } else {
            mistralrs_quant::ReplicatedLayer::from_linear(candle_nn::Linear::new(
                tok_embeddings.embeddings().clone(),
                None,
            ))?
        };

        // Determine layer range
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

        // Create RoPE embeddings for each device location
        let head_dim = cfg.head_dim();
        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|f| (f * head_dim as f32) as usize)
            .unwrap_or(head_dim);

        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<dyn PositionEncoding>> =
            HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                let rope: Arc<dyn PositionEncoding> = if rotary_dim < head_dim {
                    Arc::new(PartialRotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        rotary_dim,
                        cfg.max_position_embeddings,
                        layer_device,
                        false, // GLM-4 uses interleaved (rope_i)
                        DType::F32,
                    )?)
                } else {
                    Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        head_dim,
                        cfg.max_position_embeddings,
                        layer_device,
                        false, // GLM-4 uses interleaved (rope_i)
                        DType::F32,
                    )?)
                };
                e.insert(rope);
            }
        }

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_loaded_layers);
        let vb_l = vb_m.pp("layers");
        let attention_bias = cfg.attention_bias.unwrap_or(false);

        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rope = ropes
                .get(&layer_device.location())
                .expect("No RoPE for device location!")
                .clone();

            let vb_layer = vb_l.pp(layer_idx);
            let vb_attn = vb_layer.pp("self_attn");
            let vb_mlp = vb_layer.pp("mlp");

            // Load attention projections
            let q_proj = if attention_bias {
                mistralrs_quant::linear(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("q_proj"),
                )?
            } else {
                mistralrs_quant::linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("q_proj"),
                )?
            };
            let k_proj = if attention_bias {
                mistralrs_quant::linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("k_proj"),
                )?
            } else {
                mistralrs_quant::linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("k_proj"),
                )?
            };
            let v_proj = if attention_bias {
                mistralrs_quant::linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("v_proj"),
                )?
            } else {
                mistralrs_quant::linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    &quant_cfg,
                    vb_attn.pp("v_proj"),
                )?
            };
            let o_proj = mistralrs_quant::linear_no_bias(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                &quant_cfg,
                vb_attn.pp("o_proj"),
            )?;

            // Load MLP
            let gate_proj = mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                &quant_cfg,
                vb_mlp.pp("gate_proj"),
            )?;
            let up_proj = mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                &quant_cfg,
                vb_mlp.pp("up_proj"),
            )?;
            let down_proj = mistralrs_quant::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                &quant_cfg,
                vb_mlp.pp("down_proj"),
            )?;

            // Load all 4 normalization layers
            let input_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("input_layernorm"),
            )?;
            let post_self_attn_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("post_self_attn_layernorm"),
            )?;
            let post_attention_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("post_attention_layernorm"),
            )?;
            let post_mlp_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_layer.pp("post_mlp_layernorm"),
            )?;

            // Build attention
            let mut attn_config = AttentionConfig::new(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
            );
            if let Some(window) = cfg.sliding_window {
                attn_config = attn_config.with_sliding_window(window);
            }

            let mut attention =
                CausalAttention::new(attn_config, q_proj, k_proj, v_proj, o_proj, rope);
            if let AttentionImplementation::PagedAttention = attention_mechanism {
                attention =
                    attention.with_paged_attn(PagedAttention::new(head_dim, layer_device, None)?);
            }
            attention = attention.with_attn_dtype(dtype);

            // Build MLP
            let mlp = Mlp::from_weights(gate_proj, up_proj, down_proj, cfg.hidden_act);

            // Assemble layer
            layers.push(Glm4TransformerBlock::new(
                input_layernorm,
                attention,
                post_self_attn_layernorm,
                post_attention_layernorm,
                mlp,
                post_mlp_layernorm,
            ));
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            mapper: Some(mapper),
            dtype,
            kv_dim: cfg.head_dim() * cfg.num_key_value_heads,
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
            meta.get("general.architecture")
                .cloned()
                .try_value_into()?
        };
        if arch != "glm4" && arch != "chatglm" {
            candle_core::bail!("Expected `glm4` or `chatglm` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Get partial rotary factor if present
        let partial_rotary_factor: Option<f32> = metadata
            .get_value::<f32>("rope.partial_rotary_factor")
            .ok();

        let rotary_dim = partial_rotary_factor
            .map(|f| (f * config.head_dim as f32) as usize)
            .unwrap_or(config.head_dim);

        // Create weight source wrapper
        let mut weights = GgufWeightSource::new(&mut ct);
        let naming = GgufNaming;
        let glm4_naming = Glm4GgufNaming;

        // Load embedding weights
        let tok_embeddings = weights.load_embedding(
            &naming.token_embd(),
            config.vocab_size,
            config.hidden_size,
            device,
        )?;

        // Load output norm
        let norm = weights.load_rms_norm(&naming.output_norm(), config.rms_norm_eps, device)?;

        // Load output weights (tie to embeddings if not present)
        let output = if weights.has_tensor(&naming.output()) {
            weights.load_linear(&naming.output(), config.hidden_size, config.vocab_size, device)?
        } else {
            weights.load_linear(
                &naming.token_embd(),
                config.hidden_size,
                config.vocab_size,
                device,
            )?
        };

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
        let mut ropes: HashMap<candle_core::DeviceLocation, Arc<dyn PositionEncoding>> =
            HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry(layer_device.location())
            {
                let rope: Arc<dyn PositionEncoding> = if rotary_dim < config.head_dim {
                    Arc::new(PartialRotaryEmbedding::new(
                        config.rope_theta,
                        rotary_dim,
                        config.max_seq_len,
                        layer_device,
                        false, // GLM-4 uses interleaved (rope_i)
                        DType::F32,
                    )?)
                } else {
                    Arc::new(RotaryEmbedding::new(
                        config.rope_theta,
                        config.head_dim,
                        config.max_seq_len,
                        layer_device,
                        false, // GLM-4 uses interleaved (rope_i)
                        DType::F32,
                    )?)
                };
                e.insert(rope);
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
            let rope = ropes
                .get(&layer_device.location())
                .expect("No RoPE for device location!")
                .clone();

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
            let gate_proj = weights.load_linear(
                &naming.ffn_gate(layer_idx),
                config.hidden_size,
                config.intermediate_size,
                layer_device,
            )?;
            let up_proj = weights.load_linear(
                &naming.ffn_up(layer_idx),
                config.hidden_size,
                config.intermediate_size,
                layer_device,
            )?;
            let down_proj = weights.load_linear(
                &naming.ffn_down(layer_idx),
                config.intermediate_size,
                config.hidden_size,
                layer_device,
            )?;

            // Load all 4 normalization layers (GLM-4 specific naming)
            let input_layernorm = weights.load_rms_norm(
                &glm4_naming.input_layernorm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;
            let post_self_attn_layernorm = weights.load_rms_norm(
                &glm4_naming.post_self_attn_layernorm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;
            let post_attention_layernorm = weights.load_rms_norm(
                &glm4_naming.post_attention_layernorm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;
            let post_mlp_layernorm = weights.load_rms_norm(
                &glm4_naming.post_mlp_layernorm(layer_idx),
                config.rms_norm_eps,
                layer_device,
            )?;

            // Build attention
            let mut attn_config = AttentionConfig::new(
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
            );
            if let Some(window) = config.sliding_window {
                attn_config = attn_config.with_sliding_window(window);
            }

            let mut attention =
                CausalAttention::new(attn_config, q_proj, k_proj, v_proj, o_proj, rope);
            if let AttentionImplementation::PagedAttention = attention_mechanism {
                attention = attention
                    .with_paged_attn(PagedAttention::new(config.head_dim, layer_device, None)?);
            }
            attention = attention.with_attn_dtype(dtype);

            // Build MLP
            let mlp = Mlp::from_weights(gate_proj, up_proj, down_proj, config.hidden_act);

            // Assemble layer
            layers.push(Glm4TransformerBlock::new(
                input_layernorm,
                attention,
                post_self_attn_layernorm,
                post_attention_layernorm,
                mlp,
                post_mlp_layernorm,
            ));
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            mapper: Some(mapper),
            dtype,
            kv_dim: config.head_dim * config.num_kv_heads,
        })
    }
}

// =============================================================================
// GLM-4 GGUF Tensor Naming
// =============================================================================

/// GLM-4 specific GGUF tensor naming for the 4 normalization layers.
struct Glm4GgufNaming;

impl Glm4GgufNaming {
    fn input_layernorm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_norm.weight")
    }

    fn post_self_attn_layernorm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.attn_norm_2.weight")
    }

    fn post_attention_layernorm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_norm.weight")
    }

    fn post_mlp_layernorm(&self, layer_idx: usize) -> String {
        format!("blk.{layer_idx}.ffn_norm_2.weight")
    }
}

// =============================================================================
// Helper Methods
// =============================================================================

impl ModelWeights {
    /// Number of transformer layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// =============================================================================
// Model Trait Implementations
// =============================================================================

impl ModelTrait for ModelWeights {
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
        standard_embed(self, tokens)
    }

    fn transform(
        &self,
        hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let start_offsets: Vec<usize> = vec![ctx.position_offset];

        // Compute causal mask
        use crate::layers::CausalMasker;
        use crate::layers_masker::PastKvLenCache;
        let mask = CausalMasker.make_causal_mask_as(
            seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;

        // Skip mask for non-first chunks in paged attention
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Run through layers
        let metadata = ctx
            .paged_attn
            .as_ref()
            .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));

        let mut x = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            // Apply device mapping if present
            if let Some(mapper) = &self.mapper {
                x = mapper.map(x, i)?;
            }

            let layer_metadata = metadata
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta));

            x = layer.forward(x, mask.as_ref(), &start_offsets, &mut cache[i], layer_metadata)?;
        }

        Ok(x)
    }
}

impl TransformerModelExt for ModelWeights {
    type Layer = Glm4TransformerBlock;
    type Norm = RmsNorm;

    fn tok_embeddings(&self) -> &Embedding {
        &self.tok_embeddings
    }

    fn layers(&self) -> &[Self::Layer] {
        &self.layers
    }

    fn output_norm(&self) -> &Self::Norm {
        &self.norm
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

impl LanguageModel<[KvCache]> for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        crate::models::standard_lm_head(self, hidden)
    }
}

impl LanguageModelExt for ModelWeights {
    fn output(&self) -> &Arc<dyn QuantMethod> {
        &self.output
    }
}
