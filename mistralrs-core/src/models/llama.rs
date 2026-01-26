#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Unified Llama model implementation.
//!
//! This module provides a generic `Llama<P>` struct parameterized by position encoding.
//! Type aliases provide concrete model types:
//!
//! - `LlamaModel` = `Llama<RotaryEmbedding>` - Basic Llama (GGUF)
//! - `Llama3Model` = `Llama<Llama3RotaryEmbedding>` - Llama 3 with rope scaling

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, GgufMatMul, QuantMethod, QuantMethodConfig, QuantizedConfig,
    ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::MlpLayer,
    attention::{PositionEncoding, SdpaParams},
    device_map::DeviceMapper,
    gguf::Content,
    layers::{
        embedding, Activation, CausalMasker, Llama3RopeConfig, Llama3RotaryEmbedding, MatMul, Mlp,
        RmsNorm, RotaryEmbedding, Sdpa,
    },
    layers_masker::PastKvLenCache,
    models::{LanguageModel, LlamaConfig, Model, TransformContext, TransformerModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        IsqModel, KvCache, NormalLoadingMetadata,
    },
    serde_default_fn,
    utils::{
        gguf_metadata::ContentMetadata,
        model_config as ModelConfig,
        progress::NiceProgressBar,
        unvarbuilder::UnVarBuilder,
    },
};

use super::quantized_llama::PropsGGUF;

serde_default_fn!(bool, word_emb_default, false);

// =============================================================================
// Type Aliases for Concrete Model Types
// =============================================================================

/// Basic Llama model (Llama 2, etc.) - uses standard rotary embeddings.
pub type LlamaModel = Llama<RotaryEmbedding>;

/// Llama 3 model - uses Llama3 rotary embeddings with rope scaling support.
pub type Llama3Model = Llama<Llama3RotaryEmbedding>;

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl LlamaConfig for Config {
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
        self.rope_theta
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
}

// =============================================================================
// Attention (Generic over Position Encoding)
// =============================================================================

struct CausalSelfAttention<P: PositionEncoding> {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<P>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl<P: PositionEncoding> CausalSelfAttention<P> {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
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
                        attention_mask.clone().as_ref(),
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
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// =============================================================================
// Transformer Block (Generic over Position Encoding)
// =============================================================================

struct Block<P: PositionEncoding> {
    rms_1: RmsNorm,
    attn: CausalSelfAttention<P>,
    rms_2: RmsNorm,
    mlp: Box<dyn MlpLayer>,
}

impl<P: PositionEncoding> Block<P> {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

// =============================================================================
// Llama Model (Generic over Position Encoding)
// =============================================================================

/// Generic Llama model parameterized by position encoding type.
///
/// Use the type aliases for concrete models:
/// - `LlamaModel` for basic Llama (GGUF)
/// - `Llama3Model` for Llama 3 with rope scaling
pub struct Llama<P: PositionEncoding> {
    wte: Embedding,
    blocks: Vec<Block<P>>,
    ln_f: Option<RmsNorm>,
    lm_head: Option<Arc<dyn QuantMethod>>,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_layers: usize,
    max_seq_len: usize,
}

// =============================================================================
// Core Model Traits (Generic implementations)
// =============================================================================

impl<P: PositionEncoding + Send + Sync> Model for Llama<P> {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl<P: PositionEncoding + Send + Sync> TransformerModel for Llama<P> {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.wte.forward(tokens)
    }

    fn transform(&self, mut hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
        let start_offsets: Vec<usize> = vec![ctx.position_offset];

        // Compute mask using position offsets
        let mask = CausalMasker.make_causal_mask_as(
            ctx.seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            hidden.dtype(),
        )?;
        // PagedAttention prompt chunking filter
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Build FlashParams for this transform
        let flash_params = FlashParams {
            max_q: ctx.seq_len as u32,
            max_k: (ctx.position_offset + ctx.seq_len) as u32,
            cumulative_seqlens_q: std::collections::HashMap::new(),
            cumulative_seqlens_k: std::collections::HashMap::new(),
            causal: true,
        };

        // Run through transformer blocks
        for (block_idx, block) in self.blocks.iter().enumerate() {
            hidden = self.mapper.map(hidden, block_idx)?;
            hidden = block.forward(
                &hidden,
                &mask.clone().map(|m| m.to_device(hidden.device()).unwrap()),
                &start_offsets,
                &mut cache[block_idx],
                ctx.paged_attn
                    .as_ref()
                    .map(|pa| (pa.kv_cache[block_idx].clone(), pa.metadata)),
                &flash_params,
            )?;
        }

        Ok(hidden)
    }
}

impl<P: PositionEncoding + Send + Sync> LanguageModel for Llama<P> {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        let x = hidden.to_device(&self.device)?;
        let ln_f = self.ln_f.as_ref().expect("lm_head called but no final norm (non-tail stage?)");
        let lm_head = self.lm_head.as_ref().expect("lm_head called but no lm_head (non-tail stage?)");

        let mut x = ln_f.forward(&x)?;
        if let Some(t) = lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        MatMul.qmethod_matmul(&x, &**lm_head)
    }
}

// =============================================================================
// Helper methods
// =============================================================================

impl<P: PositionEncoding + Send + Sync> Llama<P> {
    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }

    /// Forward pass starting from pre-computed embeddings.
    ///
    /// Used by vision-language models where embeddings are modified (e.g., image features merged).
    /// Cache is passed in - the model is stateless.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        _input_ids: &Tensor, // For compatibility, not used
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        // Build transform context
        let seq_len = input_embeds.dim(1)?;
        let position_offset = seqlen_offsets.first().copied().unwrap_or(0);

        let paged_attn_ctx = metadata.map(|(kv_cache, meta)| {
            crate::models::PagedAttentionContext {
                kv_cache,
                metadata: meta,
            }
        });

        let ctx = TransformContext {
            seq_len,
            position_offset,
            paged_attn: paged_attn_ctx.as_ref(),
        };

        // Transform through layers
        let hidden = self.transform(input_embeds, &ctx, cache)?;

        // Apply lm_head
        self.lm_head(hidden)
    }

    /// Get MLP layers for ISQ/AnyMoE.
    pub fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.blocks.iter().map(|b| &*b.mlp as &dyn MlpLayer).collect()
    }

    /// Get mutable MLP layers for AnyMoE.
    pub fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.blocks.iter_mut().map(|b| &mut b.mlp).collect()
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        if let Some(ref ln_f) = self.ln_f {
            uvb_m.pp("norm").add(ln_f);
        }

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }

    /// Get model config metadata.
    pub fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

// =============================================================================
// Loading from GGUF (LlamaModel = Llama<RotaryEmbedding>)
// =============================================================================

impl ModelConfig::FromGGUF for LlamaModel {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
    ) -> Result<Self> {
        // Extract configuration from GGUF metadata
        let metadata = ContentMetadata {
            path_prefix: "llama",
            metadata: ct.get_metadata(),
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for pipeline parallelism
        let total_layers = props.num_layers();
        let layer_range = layer_range.unwrap_or(0..total_layers);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(total_layers);
        let is_last_stage = layer_end >= total_layers;

        if layer_start > 0 || layer_end < total_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                total_layers
            );
        }

        // Load embeddings
        let embed_weight = ct.tensor("token_embd.weight", device)?;
        let wte = Embedding::new(
            embed_weight.dequantize(device)?,
            props.hidden_size(),
        );

        // Load final norm and LM head only for last stage
        let ln_f = if is_last_stage {
            Some(RmsNorm::from_qtensor(
                ct.tensor("output_norm.weight", device)?,
                props.rms_norm_eps() as f32,
            )?)
        } else {
            None
        };

        let lm_head = if is_last_stage {
            let weight = if ct.has_tensor("output.weight") {
                ct.tensor("output.weight", device)?
            } else {
                // Tied embeddings
                ct.tensor("token_embd.weight", device)?
            };
            Some(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(weight),
                b: None,
            })?) as Arc<dyn QuantMethod>)
        } else {
            None
        };

        // Create rotary embeddings for each device
        let head_dim = props.head_dim();
        let mut rotary_embeddings: HashMap<_, Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);
            rotary_embeddings.entry(layer_device.location()).or_insert_with(|| {
                Arc::new(RotaryEmbedding::new_partial(
                    props.rope_theta(),
                    props.rope_dim(),
                    props.max_seq_len(),
                    layer_device,
                    false,
                    dtype,
                ).expect("Failed to create rotary embedding"))
            });
        }

        // Load transformer layers
        let mut blocks = Vec::with_capacity(layer_end - layer_start);

        for layer_idx in layer_start..layer_end {
            let prefix = format!("blk.{layer_idx}");
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);

            let rotary = rotary_embeddings
                .get(&layer_device.location())
                .expect("Missing rotary embedding for device")
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
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.attn_output.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;

            // Load MLP weights
            let gate_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_gate.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            let up_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_up.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            let down_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(ct.tensor(&format!("{prefix}.ffn_down.weight"), layer_device)?),
                b: None,
            })?) as Arc<dyn QuantMethod>;

            // Load layer norms
            let rms_1 = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), layer_device)?,
                props.rms_norm_eps() as f32,
            )?;
            let rms_2 = RmsNorm::from_qtensor(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), layer_device)?,
                props.rms_norm_eps() as f32,
            )?;

            // Create paged attention if needed
            let paged_attn = match attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, layer_device, None)?)
                }
            };

            // Create MLP
            let mlp = Mlp::from_weights(gate_proj, up_proj, down_proj, props.hidden_act());

            blocks.push(Block {
                rms_1,
                attn: CausalSelfAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_attention_heads: props.num_attention_heads(),
                    num_key_value_heads: props.num_key_value_heads(),
                    head_dim,
                    rotary_emb: rotary,
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: props.num_attention_heads() / props.num_key_value_heads(),
                        softcap: None,
                        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                        sliding_window: None,
                    },
                },
                rms_2,
                mlp: Box::new(mlp),
            });
        }

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            device: device.clone(),
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: props.max_seq_len(),
                num_layers: total_layers,
                hidden_size: props.hidden_size(),
                num_kv_heads: props.num_key_value_heads(),
                num_attn_heads: props.num_attention_heads(),
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
            },
            num_layers: total_layers,
            max_seq_len: props.max_seq_len(),
        })
    }
}

// =============================================================================
// Loading from Safetensors (Llama3Model = Llama<Llama3RotaryEmbedding>)
// =============================================================================

impl Llama3Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
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

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
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
                mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut ropes: HashMap<_, Arc<Llama3RotaryEmbedding>> = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(Llama3RotaryEmbedding::new_llama3(
                    vb_m.dtype(),
                    cfg,
                    device,
                    is_gptx,
                ).expect("Failed to create Llama3 rotary embedding"))
            });
        }
        let blocks: Vec<_> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
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
            let comm = mapper.get_comm_for(i)?;

            // Load attention
            let vb_attn = mapper.set_device(i, vb_m.pp(format!("layers.{i}.self_attn")), normal_loading_metadata.loading_isq);
            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = ColumnParallelLayer::new(
                size_in,
                size_q,
                &cfg.quantization_config,
                false,
                &comm,
                vb_attn.pp("q_proj"),
            )?;
            let kv_shard = mistralrs_quant::compute_kv_shard(
                cfg.num_key_value_heads,
                cfg.hidden_size / cfg.num_attention_heads,
                &comm,
            );
            let k_proj = ColumnParallelLayer::new_with_shard(
                size_in,
                size_kv,
                &cfg.quantization_config,
                false,
                &comm,
                kv_shard,
                vb_attn.pp("k_proj"),
            )?;
            let v_proj = ColumnParallelLayer::new_with_shard(
                size_in,
                size_kv,
                &cfg.quantization_config,
                false,
                &comm,
                kv_shard,
                vb_attn.pp("v_proj"),
            )?;
            let o_proj = RowParallelLayer::new(
                size_q,
                size_in,
                &cfg.quantization_config,
                false,
                &comm,
                vb_attn.pp("o_proj"),
            )?;

            // Load MLP
            let mlp = Mlp::new(
                mapper.set_device(i, vb_m.pp(format!("layers.{i}.mlp")), normal_loading_metadata.loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                &comm,
            )?;

            // Load layer norms
            let rms_1 = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_m.pp(format!("layers.{i}.input_layernorm")), false),
            )?;
            let rms_2 = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_m.pp(format!("layers.{i}.post_attention_layernorm")), false),
            )?;

            Ok(Block {
                rms_1,
                attn: CausalSelfAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_attention_heads: cfg.num_attention_heads / comm.world_size(),
                    num_key_value_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
                    head_dim,
                    rotary_emb,
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
                },
                rms_2,
                mlp: Box::new(mlp),
            })
        })?;

        Ok(Self {
            wte,
            blocks,
            ln_f: Some(ln_f),
            lm_head: Some(lm_head),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            },
            mapper,
            num_layers: cfg.num_hidden_layers,
            max_seq_len: cfg.max_position_embeddings,
        })
    }
}

// =============================================================================
// ISQ Support
// =============================================================================

impl<P: PositionEncoding + Send + Sync> IsqModel for Llama<P> {
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
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
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
        self.residual_tensors_m(uvb.pp("model"))
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        // lm_head (if present)
        if self.lm_head.is_some() {
            names.push(None);
        }
        for i in 0..self.blocks.len() {
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
