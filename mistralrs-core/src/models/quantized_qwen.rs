#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::QuantMethod;

use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, RmsNorm, RmsNormQkNorm};
use crate::layers_masker::PastKvLenCache;
use crate::models::{LanguageModel, Model, TransformContext, TransformerModel};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::{
    load_transformer_layers, GgufNaming, GgufWeightSource, StandardTransformerBlock,
    TensorNaming, TransformerConfig, WeightSource,
};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

/// Qwen2 model weights using the generic transformer builder infrastructure.
///
/// The model uses pre-norm architecture with:
/// - RMS normalization for attention and FFN
/// - Optional Q/K normalization (when present, e.g., Qwen3 models loaded via Qwen2 path)
/// - Optional Q/K/V attention biases (Qwen2-specific)
/// - Gated MLP with SiLU activation
/// - RoPE positional embeddings
///
/// This implementation supports both "qwen2" and "qwen3" GGUF architectures.
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub max_seq_len: usize,
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
        // Verify architecture (accept both qwen2 and qwen3)
        let meta = ct.get_metadata();
        let arch: String = {
            use crate::utils::gguf_metadata::TryValueInto;
            meta.get("general.architecture")
                .cloned()
                .try_value_into()?
        };
        if arch != "qwen2" && arch != "qwen3" {
            candle_core::bail!("Expected `qwen2` or `qwen3` architecture, got `{arch}`.");
        }

        // Parse config from GGUF metadata using generic infrastructure
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let config = TransformerConfig::from_gguf_metadata(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            // Qwen2 has optional Q/K/V biases (but not O bias)
            .with_attention_bias();

        // Create weight source wrapper
        let mut weights = GgufWeightSource::new(&mut ct);
        let naming = GgufNaming;

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
            weights.load_linear(&naming.output(), device)?
        } else {
            weights.load_linear(&naming.token_embd(), device)?
        };

        // Load transformer layers using generic infrastructure
        // Qwen2 may have optional Q/K norm (when loading qwen3 files through qwen2 path)
        let layers = load_transformer_layers(
            &config,
            &mut weights,
            &naming,
            layer_range,
            &*mapper,
            device,
            attention_mechanism,
            dtype,
            |ctx, builder, weights| {
                // Check if Q/K norm exists (Qwen3-style) and load if present
                let q_norm_name = naming.attn_q_norm(ctx.layer_idx);
                let k_norm_name = naming.attn_k_norm(ctx.layer_idx);

                if weights.has_tensor(&q_norm_name) && weights.has_tensor(&k_norm_name) {
                    let q_norm = weights.load_rms_norm(&q_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    let k_norm = weights.load_rms_norm(&k_norm_name, ctx.rms_norm_eps, ctx.device)?;
                    let qk_norm: Arc<dyn crate::attention::QkNorm> =
                        Arc::new(RmsNormQkNorm::new(q_norm, k_norm));
                    Ok(builder.with_qk_norm(qk_norm))
                } else {
                    // No Q/K norm (standard Qwen2)
                    Ok(builder)
                }
            },
        )?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    /// Run transformer layers on hidden states with the given context.
    ///
    /// This is the core layer iteration logic used by all forward methods.
    /// The cache is passed as a parameter to avoid borrow checker issues.
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

            let layer_metadata = metadata
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta));

            hidden = layer.forward(
                hidden,
                mask,
                position_offsets,
                &mut cache[i],
                layer_metadata,
            )?;
        }

        Ok(hidden)
    }

    /// Forward pass for embeddings - returns hidden states before LM head.
    /// Used by GGUF embedding pipeline.
    pub fn forward_hidden_states(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let embeds = self.embed(x)?;
        let ctx = TransformContext {
            seq_len: embeds.dim(1)?,
            position_offset: start_offsets.first().copied().unwrap_or(0),
            paged_attn: None,
        };
        let hidden = self.transform(embeds, &ctx, cache)?;
        // Return hidden states after final norm (before output projection)
        self.norm.forward(&hidden)
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

        // Compute mask using position offsets
        let mask = CausalMasker.make_causal_mask_as(
            seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;
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
        let x = self.norm.forward(&hidden)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }
}
