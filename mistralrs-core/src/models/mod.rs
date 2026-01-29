// Models are parameterized by their inference state type S.
// See TokenizerModel<S> for the core trait hierarchy.

pub(crate) mod deepseek2;
pub(crate) mod deepseek3;
pub(crate) mod gemma;
pub(crate) mod gemma2;
pub(crate) mod glm4;
pub(crate) mod quantized_gemma;
pub(crate) mod quantized_gemma2;
pub(crate) mod gpt_oss;
pub(crate) mod granite;
pub(crate) mod llama;
pub(crate) mod mistral;
pub(crate) mod mixtral;
pub(crate) mod quantized_mistral;
pub(crate) mod phi2;
pub(crate) mod phi3;
pub(crate) mod phi3_5_moe;
pub(crate) mod quantized_llama;
pub(crate) mod quantized_mistral3;
pub(crate) mod quantized_phi2;
pub(crate) mod quantized_phi3;
pub(crate) mod quantized_qwen;
pub(crate) mod quantized_qwen3;
pub(crate) mod quantized_qwen3_moe;
pub(crate) mod quantized_starcoder2;
pub(crate) mod qwen2;
pub(crate) mod qwen3;
pub(crate) mod qwen3_moe;
pub(crate) mod smollm3;
pub(crate) mod starcoder2;

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::QuantMethod;

use crate::device_map::DeviceMapper;
pub use crate::kv_cache::InferenceState;
use crate::kv_cache::CacheLayout;
use crate::layers::{CausalMasker, MatMul};
use crate::layers_masker::PastKvLenCache;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::KvCache;

/// Base trait for all models.
///
/// Every model has weights that live on a device. This is the fundamental
/// property shared by all model types: transformers, diffusion, vision, embedding, etc.
pub trait Model: Send + Sync {
    /// The device where model weights reside.
    fn device(&self) -> &Device;
}

/// Context for transformer forward pass.
///
/// Contains all information needed for a single forward pass through transformer layers.
/// Separates concerns: the model receives what it needs, pipeline handles orchestration.
pub struct TransformContext<'a> {
    /// Sequence length being processed (for causal mask computation).
    /// Derived from input tensor shape, not from token count.
    pub seq_len: usize,

    /// Position offset for RoPE (cumulative tokens already in KV cache).
    pub position_offset: usize,

    /// PagedAttention context, if using paged attention.
    /// None when using eager attention or during prefill without paging.
    pub paged_attn: Option<&'a PagedAttentionContext<'a>>,

    /// Flash attention parameters, if using flash attention.
    /// None when using standard SDPA.
    pub flash_params: Option<&'a FlashParams>,

    /// Explicit position IDs for each token in the sequence.
    /// Some models (like Phi3) use explicit position IDs instead of deriving from offset.
    /// When None, models should derive positions from `position_offset`.
    pub position_ids: Option<&'a [usize]>,
}

/// PagedAttention-specific context for transform.
pub struct PagedAttentionContext<'a> {
    /// KV cache tensors per layer: Vec<(key_cache, value_cache)>
    pub kv_cache: Vec<(Tensor, Tensor)>,

    /// PagedAttention metadata (block tables, slot mappings, etc.)
    pub metadata: &'a PagedAttentionInputMetadata,
}

/// Trait for transformer layers (used as associated type bound).
///
/// Transformer layers take hidden states and produce transformed hidden states.
/// This trait is implemented by `TransformerBlock<N, A, F>` and its type aliases.
pub trait TransformerLayer: Send + Sync {
    /// Forward pass through this layer.
    fn forward(
        &self,
        hidden: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        cache: &mut KvCache,
        paged_attn_meta: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor>;
}

// =============================================================================
// Model Trait Hierarchy (State-Parameterized)
// =============================================================================
//
// Models are parameterized by their inference state type S:
// - [KvCache]     - standard attention-only models (Llama, Mistral, Qwen, etc.)
// - HybridCache   - mixed attention/SSM models (Granite, Jamba)
//
// The state type captures the architecture. The trait captures the capability.
//
// Hierarchy:
//   Model                        - base (weights on a device)
//   └── TokenizerModel<S>        - embed + transform (processes token input)
//       ├── LanguageModel<S>     - adds lm_head (text generation)
//       └── EmbeddingModel<S>    - adds pool (dense vector output)

/// Trait for models that process token input.
///
/// A `TokenizerModel` can embed token IDs into continuous representations
/// and transform them through layers with inference state type S.
///
/// # Type Parameter
///
/// - `S`: The inference state type. Captures the model's architecture:
///   - `[KvCache]` - standard attention models (Llama, Mistral, Qwen, etc.)
///   - `HybridCache` - mixed attention/SSM models (Granite, Jamba)
///
/// # Object Safety
///
/// This trait is object-safe when S is concrete: `dyn TokenizerModel<[KvCache]>`.
pub trait TokenizerModel<S: InferenceState + ?Sized>: Model {
    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize;

    /// Maximum sequence length this model supports.
    fn max_seq_len(&self) -> usize;

    /// KV cache dimension (typically `head_dim * num_kv_heads`).
    ///
    /// Used by the default `cache_layout()` implementation.
    fn kv_dim(&self) -> usize;

    /// Convert token IDs to embeddings.
    ///
    /// Input: token IDs [batch, seq_len]
    /// Output: embeddings [batch, seq_len, hidden_dim]
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;

    /// Transform hidden states through model layers.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: hidden states [batch, seq_len, hidden_dim]
    fn transform(
        &self,
        hidden: Tensor,
        ctx: &TransformContext,
        state: &mut S,
    ) -> Result<Tensor>;

    /// Describes cache requirements for this model.
    ///
    /// Default: uniform KV cache for all layers. Override for:
    /// - Hybrid models (mixed attention/Mamba): `CacheLayout::hybrid(...)`
    /// - Sliding window attention: `CacheLayout::sliding_window(...)`
    /// - Attention sinks: `CacheLayout::with_sinks(...)`
    fn cache_layout(&self) -> CacheLayout {
        CacheLayout::uniform_kv(self.num_layers(), self.kv_dim(), self.max_seq_len())
    }
}

/// Language model: adds next-token prediction to TokenizerModel.
///
/// The pipeline (not the model) determines which methods to call based on its stage:
/// - HEAD: `embed()` → `transform()` → send activation
/// - TAIL: receive → `transform()` → `lm_head()`
pub trait LanguageModel<S: InferenceState + ?Sized>: TokenizerModel<S> {
    /// Project hidden states to vocabulary logits.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: logits [batch, seq_len, vocab_size]
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}

/// Embedding model: adds pooling to TokenizerModel.
///
/// Embedding models transform input tokens to dense vectors without
/// a language modeling head.
pub trait EmbeddingModel<S: InferenceState + ?Sized>: TokenizerModel<S> {
    /// Pool hidden states to produce embeddings.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: embeddings [batch, embedding_dim]
    fn pool(&self, hidden: Tensor) -> Result<Tensor>;
}

// =============================================================================
// Extension Traits with Associated Types (for typed pipelines)
// =============================================================================

/// Extension trait providing accessors for transformer model components.
///
/// This trait is **not object-safe** due to associated types. It's used by typed
/// pipelines like `TextPipeline<M>` where the concrete model type is known at
/// compile time.
///
/// Models implementing this trait can use the helper functions for standard
/// implementations of `TransformerModel` methods.
pub trait TransformerModelExt: TokenizerModel<[KvCache]> {
    /// The transformer layer type for this model.
    type Layer: TransformerLayer;

    /// The output normalization type for this model.
    type Norm: Module;

    /// Access to token embeddings.
    fn tok_embeddings(&self) -> &Embedding;

    /// Access to transformer layers.
    fn layers(&self) -> &[Self::Layer];

    /// Access to output normalization layer.
    fn output_norm(&self) -> &Self::Norm;

    /// Access to device mapper (if any).
    fn mapper(&self) -> Option<&dyn DeviceMapper>;

    /// The dtype used for attention computation.
    fn model_dtype(&self) -> DType;
}

/// Extension trait providing accessor for the output projection.
///
/// This trait is **not object-safe** due to the `TransformerModelExt` bound.
/// Used by typed pipelines for models with standard LM head structure.
pub trait LanguageModelExt: LanguageModel<[KvCache]> + TransformerModelExt {
    /// Access to output projection weights.
    fn output(&self) -> &Arc<dyn QuantMethod>;
}

// =============================================================================
// Helper Functions for Standard Implementations
// =============================================================================

/// Standard implementation of `embed` for models with `TransformerModelExt`.
///
/// Simply forwards tokens through the token embeddings layer.
pub fn standard_embed<M: TransformerModelExt + ?Sized>(
    model: &M,
    tokens: &Tensor,
) -> Result<Tensor> {
    model.tok_embeddings().forward(tokens)
}

/// Standard implementation of `transform` for models with `TransformerModelExt`.
///
/// Creates a causal mask and runs through all transformer layers with device mapping.
pub fn standard_transform<M: TransformerModelExt + ?Sized>(
    model: &M,
    hidden: Tensor,
    ctx: &TransformContext,
    cache: &mut [KvCache],
) -> Result<Tensor> {
    let seq_len = hidden.dim(1)?;
    let start_offsets: Vec<usize> = vec![ctx.position_offset];

    // Compute causal mask
    let mask = CausalMasker.make_causal_mask_as(
        seq_len,
        hidden.device(),
        &start_offsets.as_slice() as &dyn PastKvLenCache,
        model.model_dtype(),
    )?;

    // Skip mask for non-first chunks in paged attention
    let mask = mask.filter(|_| {
        ctx.paged_attn
            .as_ref()
            .map(|pa| pa.metadata.is_first_prompt_chunk)
            .unwrap_or(true)
    });

    // Run through layers
    standard_run_layers(model, hidden, mask.as_ref(), &start_offsets, ctx, cache)
}

/// Standard implementation of layer iteration for models with `TransformerModelExt`.
///
/// Iterates through layers with device mapping and paged attention support.
pub fn standard_run_layers<M: TransformerModelExt + ?Sized>(
    model: &M,
    mut hidden: Tensor,
    mask: Option<&Tensor>,
    position_offsets: &[usize],
    ctx: &TransformContext,
    cache: &mut [KvCache],
) -> Result<Tensor> {
    let metadata = ctx
        .paged_attn
        .as_ref()
        .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));

    for (i, layer) in model.layers().iter().enumerate() {
        // Apply device mapping if present
        if let Some(mapper) = model.mapper() {
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

/// Standard implementation of `lm_head` for models with `LanguageModelExt`.
///
/// Applies output normalization then projects to vocabulary size.
pub fn standard_lm_head<M: LanguageModelExt + ?Sized>(model: &M, hidden: Tensor) -> Result<Tensor> {
    let x = model.output_norm().forward(&hidden)?;
    MatMul.qmethod_matmul(&x.contiguous()?, &**model.output())
}

// =============================================================================
// Unified Model Configuration Traits
// =============================================================================
//
// These traits abstract over configuration sources (JSON config.json vs GGUF
// metadata), enabling a single model implementation to be constructed from
// either source.

use crate::layers::Activation;
use mistralrs_quant::QuantizedConfig;

/// Configuration trait for decoder-only language models.
///
/// This trait captures the hyperparameters common to all decoder-only transformers
/// used for text generation: Llama, Mistral, Qwen, Phi, etc.
///
/// Implemented by both JSON config (safetensors) and GGUF metadata, enabling
/// a single model implementation to be constructed from either source.
///
/// # What this captures
///
/// - **Decoder-only transformers** with causal attention
/// - **RoPE positional encoding** (rope_theta)
/// - **Pre-norm architecture** with RmsNorm (rms_norm_eps)
/// - **Grouped-query attention** (num_kv_heads)
/// - **Gated MLP** (intermediate_size, hidden_act)
///
/// # What this does NOT capture
///
/// - Encoder-decoder models (T5, BART)
/// - Encoder-only models (BERT)
/// - Vision/diffusion models
/// - Models with different positional encoding (learned, ALiBi)
///
/// # Example
/// ```ignore
/// fn load_model<C: LanguageModelConfig>(cfg: &C, vb: VarBuilder) -> Result<impl LanguageModel> {
///     let hidden_size = cfg.hidden_size();
///     let num_layers = cfg.num_layers();
///     // ... construct model using config accessors
/// }
/// ```
pub trait LanguageModelConfig {
    /// Hidden dimension (embedding size).
    fn hidden_size(&self) -> usize;

    /// Feed-forward intermediate dimension.
    /// Note: GGUF loading infers this from weight shapes; used by safetensors loading.
    #[allow(dead_code)]
    fn intermediate_size(&self) -> usize;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;

    /// Number of key-value heads (for GQA/MQA).
    fn num_key_value_heads(&self) -> usize;

    /// Vocabulary size.
    /// Note: GGUF loading infers this from embedding weights; used by safetensors loading.
    #[allow(dead_code)]
    fn vocab_size(&self) -> usize;

    /// RMS normalization epsilon.
    fn rms_norm_eps(&self) -> f64;

    /// RoPE theta (frequency base).
    fn rope_theta(&self) -> f32;

    /// Maximum sequence length.
    fn max_seq_len(&self) -> usize;

    /// Activation function for MLP.
    fn hidden_act(&self) -> Activation {
        Activation::Silu
    }

    /// Whether to tie input/output embeddings.
    /// Note: GGUF loading checks weight existence; used by safetensors loading.
    #[allow(dead_code)]
    fn tie_word_embeddings(&self) -> bool {
        false
    }

    /// Head dimension (derived).
    fn head_dim(&self) -> usize {
        self.hidden_size() / self.num_attention_heads()
    }

    /// RoPE dimension (usually same as head_dim).
    fn rope_dim(&self) -> usize {
        self.head_dim()
    }

    // MoE configuration (optional)

    /// Number of experts (0 for non-MoE models).
    /// Note: MoE support not yet implemented in unified Llama.
    #[allow(dead_code)]
    fn num_experts(&self) -> usize {
        0
    }

    /// Number of experts used per token.
    /// Note: MoE support not yet implemented in unified Llama.
    #[allow(dead_code)]
    fn num_experts_used(&self) -> usize {
        0
    }

    /// Quantization configuration (for quantized models).
    /// Returns None for full-precision models.
    fn quantization_config(&self) -> Option<&QuantizedConfig> {
        None
    }
}

/// Deprecated alias for backwards compatibility.
#[deprecated(since = "0.8.0", note = "Use LanguageModelConfig instead")]
pub trait LlamaConfig: LanguageModelConfig {}

// Blanket impl: anything implementing LanguageModelConfig also implements LlamaConfig
#[allow(deprecated)]
impl<T: LanguageModelConfig> LlamaConfig for T {}
