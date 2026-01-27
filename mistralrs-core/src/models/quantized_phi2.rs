#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Phi2 model using the generic transformer infrastructure.
//!
//! Uses `ParallelTransformerBlock<LayerNorm, FusedQkvCausalAttention, NonGatedMlp>` composition
//! to reduce code duplication while handling Phi2's unique architecture:
//!
//! - **Parallel attention+MLP**: Both computed from the same normalized input
//! - **Partial RoPE**: Rotary embeddings applied only to first `rope_dim` dimensions
//! - **Fused QKV projection**: Single weight for Q, K, V
//! - **LayerNorm**: Not RmsNorm
//! - **Non-gated MLP**: Uses GELU activation without gating

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::{AttentionConfig, FusedQkvCausalAttention, PositionEncoding};
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    Activation, CausalMasker, MatMul, NonGatedMlp, ParallelTransformerBlock, PartialRotaryEmbedding,
    QLinear,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{
    LanguageModel, LanguageModelExt, Model, TransformContext, TransformerModel,
    TransformerModelExt,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::KvCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// A parallel transformer block for Phi2 using fused QKV attention and non-gated MLP.
///
/// Uses the generic `ParallelTransformerBlock` with:
/// - `LayerNorm` for normalization (single norm before both attention and MLP)
/// - `FusedQkvCausalAttention` for fused QKV with partial RoPE
/// - `NonGatedMlp` for GELU activation without gating
pub type Phi2Block = ParallelTransformerBlock<LayerNorm, FusedQkvCausalAttention, NonGatedMlp>;

// phi2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rope_dim: usize,
    ln_eps: f64,
    max_seq_len: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("phi2")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
            "context_length",
        ];
        c.has_required_keys(&required)?;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            ln_eps: c.get_value::<f32>("attention.layer_norm_rms_epsilon")? as f64,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
        };

        Ok(props)
    }
}

/// Create a LayerNorm from GGUF quantized tensors.
fn layer_norm_from_gguf(
    weight: candle_core::quantized::QTensor,
    bias: candle_core::quantized::QTensor,
    eps: f64,
) -> Result<LayerNorm> {
    let w = weight.dequantize(&weight.device())?;
    let b = bias.dequantize(&bias.device())?;
    Ok(LayerNorm::new(w, b, eps))
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Phi2Block>,
    /// Final norm layer (None for non-last pipeline stages)
    output_norm: Option<LayerNorm>,
    /// Output/LM head layer (None for non-last pipeline stages)
    output: Option<Arc<dyn QuantMethod>>,
    device: Device,
    max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
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
        let metadata = ContentMetadata {
            path_prefix: "phi2",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rope_dim,
            ln_eps,
            max_seq_len,
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

        let head_dim = embedding_length / head_count;

        // Create partial rotary embedding for Phi2 (applies RoPE to first rope_dim dimensions)
        let rope = Arc::new(PartialRotaryEmbedding::new(
            10_000.0, // Phi2 uses 10k base
            rope_dim,
            max_seq_len,
            device,
            false, // Not GPT-NeoX style
            dtype,
        )?) as Arc<dyn PositionEncoding>;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;

        // PP: Only load norm and output (LM head) for last stage
        let is_last_stage = layer_end >= total_layers;
        let output_norm = if is_last_stage {
            Some(layer_norm_from_gguf(
                ct.tensor("output_norm.weight", device)?,
                ct.tensor("output_norm.bias", device)?,
                ln_eps,
            )?)
        } else {
            None
        };
        let output = if is_last_stage {
            let out = QLinear::new(&mut ct, "output", device)?;
            let QMatMul::QTensor(out_w) = out.inner_ref().clone() else {
                unreachable!()
            };
            Some(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: out_w,
                b: out.bias().cloned(),
            })?) as Arc<dyn QuantMethod>)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_loaded_layers);

        // Only load layers in the specified range
        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);

            // Load attention norm (single norm for parallel block)
            let attn_norm = layer_norm_from_gguf(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), layer_device)?,
                ct.tensor(&format!("{prefix}.attn_norm.bias"), layer_device)?,
                ln_eps,
            )?;

            // Load fused QKV projection
            let qkv = QLinear::new(&mut ct, &format!("{prefix}.attn_qkv"), layer_device)?;
            let QMatMul::QTensor(qkv_w) = qkv.inner_ref().clone() else {
                unreachable!()
            };
            let qkv_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: qkv_w,
                b: qkv.bias().cloned(),
            })?) as Arc<dyn QuantMethod>;

            // Load output projection
            let out = QLinear::new(&mut ct, &format!("{prefix}.attn_output"), layer_device)?;
            let QMatMul::QTensor(out_w) = out.inner_ref().clone() else {
                unreachable!()
            };
            let o_proj = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: out_w,
                b: out.bias().cloned(),
            })?) as Arc<dyn QuantMethod>;

            // Create attention config
            let attn_config = AttentionConfig::new(head_count, head_count_kv, head_dim);

            // Create paged attention if needed
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, layer_device, None)?)
                }
            };

            // Create fused QKV attention
            let mut attention = FusedQkvCausalAttention::new(
                attn_config,
                qkv_proj,
                o_proj,
                rope.clone(),
            )
            .with_attn_dtype(dtype);

            if let Some(pa) = paged_attn {
                attention = attention.with_paged_attn(pa);
            }

            // Load MLP weights (non-gated GELU)
            let ffn_up = QLinear::new(&mut ct, &format!("{prefix}.ffn_up"), layer_device)?;
            let ffn_down = QLinear::new(&mut ct, &format!("{prefix}.ffn_down"), layer_device)?;
            let QMatMul::QTensor(ffn_up_w) = ffn_up.inner_ref().clone() else {
                unreachable!()
            };
            let QMatMul::QTensor(ffn_down_w) = ffn_down.inner_ref().clone() else {
                unreachable!()
            };

            let mlp = NonGatedMlp::from_weights(
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_up_w,
                    b: ffn_up.bias().cloned(),
                })?),
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: ffn_down_w,
                    b: ffn_down.bias().cloned(),
                })?),
                Activation::NewGelu, // Phi2 uses GELU
            );

            // Create parallel transformer block
            layers.push(Phi2Block::new(attn_norm, attention, mlp));
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            device: device.clone(),
            max_seq_len,
            mapper: Some(mapper),
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

    fn transform(
        &self,
        hidden: Tensor,
        ctx: &TransformContext,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
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
        let metadata = ctx
            .paged_attn
            .as_ref()
            .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));

        let mut x = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
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

impl LanguageModel for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        // Move to model device and apply final norm
        let hidden = hidden.to_device(&self.device)?;
        let x = self
            .output_norm
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

// Extension trait - accessors and associated types for typed pipelines
impl TransformerModelExt for ModelWeights {
    type Layer = Phi2Block;
    type Norm = LayerNorm;

    fn tok_embeddings(&self) -> &Embedding {
        &self.tok_embeddings
    }

    fn layers(&self) -> &[Self::Layer] {
        &self.layers
    }

    fn output_norm(&self) -> &Self::Norm {
        self.output_norm
            .as_ref()
            .expect("output_norm called on non-last pipeline stage")
    }

    fn mapper(&self) -> Option<&dyn DeviceMapper> {
        self.mapper.as_ref().map(|m| m.as_ref() as &dyn DeviceMapper)
    }

    fn model_dtype(&self) -> DType {
        self.dtype
    }
}

impl LanguageModelExt for ModelWeights {
    fn output(&self) -> &Arc<dyn QuantMethod> {
        self.output
            .as_ref()
            .expect("output called on non-last pipeline stage")
    }
}
