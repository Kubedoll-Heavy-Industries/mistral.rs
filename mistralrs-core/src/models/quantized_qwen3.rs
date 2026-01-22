#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::SdpaParams;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::models::{TransformContext, TransformerModel};
use crate::layers::{CausalMasker, MatMul, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, HookContainer, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        MatMul.qmethod_matmul(&y, &*self.feed_forward_w2)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    mlp: Mlp,
    ffn_norm: QRmsNorm,
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
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    /// Starting layer index for pipeline parallelism
    layer_start: usize,
    /// Total layers in the full model (for hooks)
    total_layers: usize,
    /// Pipeline hook for distributed inference
    hook: Option<HookContainer>,
}

// qwen3 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
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
}

fn verify_qwen3_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen3" {
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
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for partial loading (pipeline parallelism)
        let layer_start = layer_range.as_ref().map(|r| r.start).unwrap_or(0);
        let layer_end = layer_range
            .as_ref()
            .map(|r| r.end.min(block_count))
            .unwrap_or(block_count);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < block_count {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                block_count
            );
        }

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
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

            let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
            let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
            let mlp = Mlp {
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

            // Qwen3 always has q_norm and k_norm
            let q_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let k_norm = QRmsNorm::new(
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
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                q_norm,
                k_norm,
                mlp,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
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
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            // Only allocate cache for loaded layers
            cache: EitherCache::Normal(NormalCache::new(num_loaded_layers, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
            layer_start,
            total_layers: block_count,
            hook: None,
        })
    }
}

impl ModelWeights {
    /// Check if this is the first pipeline stage (has embedding layer).
    pub fn is_first_stage(&self) -> bool {
        self.layer_start == 0
    }

    /// Check if this is the last pipeline stage (has lm_head).
    pub fn is_last_stage(&self) -> bool {
        self.layer_start + self.layers.len() >= self.total_layers
    }

    /// Set the pipeline hook for distributed inference.
    pub fn set_hook(&mut self, hook: HookContainer) {
        self.hook = Some(hook);
    }

    /// Get a reference to the pipeline hook.
    pub fn get_hook(&self) -> Option<&HookContainer> {
        self.hook.as_ref()
    }

    /// Forward pass for pipeline parallelism.
    ///
    /// # Arguments
    /// * `x` - Token IDs tensor
    /// * `input_activation` - Activation from previous pipeline stage (None for first stage)
    /// * `start_offsets` - Sequence position offsets for RoPE
    /// * `context_lens` - Context lengths for logit extraction
    /// * `metadata` - PagedAttention metadata if enabled
    ///
    /// # Returns
    /// * For last stage: logits tensor after lm_head
    /// * For non-last stages: hidden states tensor after transformer layers
    pub fn forward(
        &self,
        x: &Tensor,
        input_activation: Option<&Tensor>,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        // Use provided activation or compute embeddings
        let mut layer_in = match input_activation {
            Some(act) => act.clone(),
            None => self.tok_embeddings.forward(x)?,
        };

        let cache = &mut self.cache.normal().0;
        // Mask shape source:
        // - TAIL: activation tensor [batch, seq_len, hidden] → slice to [batch, seq_len]
        // - HEAD: input_ids [batch, seq_len] directly
        let mask_shape_source = match input_activation {
            Some(act) => act.i((.., .., 0usize))?, // TAIL: [batch, seq_len, hidden] → [batch, seq_len]
            None => x.clone(),               // HEAD: input_ids is already [batch, seq_len]
        };
        let mask = CausalMasker.make_causal_mask_matrix(
            &mask_shape_source,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            self.layers[0].n_head,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }

            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                mask.as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            layer_in = (x + residual)?;
        }

        // Only apply final norm + lm_head on last stage
        if self.is_last_stage() {
            let x = self.norm.forward(&layer_in)?;
            extract_logits(
                &MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)?,
                context_lens,
            )
        } else {
            // Non-last stage: return hidden states for next stage
            Ok(layer_in)
        }
    }

    // ========================================================================
    // Pipeline Parallelism Building Blocks
    // ========================================================================
    //
    // These methods decompose the forward pass into three stages:
    // 1. get_input_embeddings: tokens → embeddings (HEAD only)
    // 2. forward_layers: hidden → hidden (all stages)
    // 3. apply_lm_head: hidden → logits (TAIL only)
    //
    // For single-node inference, call forward() which composes all three.
    // For pipeline parallelism, each stage calls only its relevant methods.

    /// Convert token IDs to embeddings (HEAD stage only).
    /// Input: token IDs tensor [batch, seq_len]
    /// Output: embeddings [batch, seq_len, hidden_dim]
    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input_ids)
    }

    /// Run transformer layers on hidden states.
    ///
    /// # Arguments
    /// * `mask_shape` - Tensor with shape [batch, seq_len] for causal mask computation.
    ///                  For HEAD: use input_ids directly.
    ///                  For TAIL: use `hidden.i((.., .., 0usize))?` to get [batch, seq_len] from activation.
    /// * `hidden` - Hidden states [batch, seq_len, hidden_dim]
    /// * `start_offsets` - Sequence position offsets for RoPE
    /// * `metadata` - PagedAttention metadata if enabled
    ///
    /// # Returns
    /// Hidden states after transformer layers [batch, seq_len, hidden_dim]
    pub fn forward_layers(
        &self,
        mask_shape: &Tensor,
        mut hidden: Tensor,
        start_offsets: &[usize],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask_matrix(
            mask_shape,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            self.layers[0].n_head,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let x = hidden;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                mask.as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            hidden = (x + residual)?;
        }

        Ok(hidden)
    }

    /// Apply final layer norm and lm_head projection (TAIL stage only).
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: logits [batch, vocab_size] (last position extracted)
    pub fn apply_lm_head(
        &self,
        hidden: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let x = self.norm.forward(&hidden)?;
        extract_logits(
            &MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)?,
            context_lens,
        )
    }

    /// Forward pass for embeddings - returns hidden states before LM head.
    /// Used by GGUF embedding pipeline.
    pub fn forward_hidden_states(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
    ) -> Result<Tensor> {
        let embeds = self.get_input_embeddings(x)?;
        let hidden = self.forward_layers(x, embeds, start_offsets, None)?;
        // Return hidden states after final norm (before LM head)
        self.norm.forward(&hidden)
    }

    /// Get the attention implementation used by this model.
    pub fn attention_implementation(&self) -> AttentionImplementation {
        if self.layers.first().map(|l| l.paged_attn.is_some()).unwrap_or(false) {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        }
    }
}

// ============================================================================
// TransformerModel Implementation
// ============================================================================
//
// This impl provides the clean abstraction for pipeline parallelism.
// The pipeline calls embed/transform/lm_head based on which stage it is.

impl TransformerModel for ModelWeights {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(tokens)
    }

    fn transform(&self, mut hidden: Tensor, ctx: &TransformContext) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;

        // Derive mask shape from hidden states: [batch, seq_len, hidden] → [batch, seq_len]
        // This avoids needing input_ids for TAIL stages.
        let mask_shape = hidden.i((.., .., 0usize))?;
        let start_offsets: Vec<usize> = vec![ctx.position_offset];
        let start_offsets_slice = start_offsets.as_slice();

        // Always use start_offsets for mask computation - this is the correct RoPE position.
        // For TAIL stages, ctx.position_offset comes from HEAD via activation message.
        // Using cache would be wrong because TAIL's cache might be empty/stale.
        let mask = CausalMasker.make_causal_mask_matrix(
            &mask_shape,
            &start_offsets_slice as &dyn PastKvLenCache,
            self.dtype,
            self.layers[0].n_head,
        )?;

        // PagedAttention prompt chunking: only apply mask on first chunk
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }

            let x = hidden;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                mask.as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                &start_offsets,
                &mut cache[i],
                ctx.paged_attn
                    .as_ref()
                    .map(|pa| (pa.kv_cache[i].clone(), pa.metadata)),
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            hidden = (x + residual)?;
        }

        Ok(hidden)
    }

    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        let x = self.norm.forward(&hidden)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }

    fn has_embed(&self) -> bool {
        self.is_first_stage()
    }

    fn has_lm_head(&self) -> bool {
        self.is_last_stage()
    }

    fn attention_impl(&self) -> AttentionImplementation {
        self.attention_implementation()
    }
}
