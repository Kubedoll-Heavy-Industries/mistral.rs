pub(crate) mod deepseek2;
pub(crate) mod deepseek3;
pub(crate) mod gemma;
pub(crate) mod gemma2;
pub(crate) mod glm4;
pub(crate) mod gpt_oss;
pub(crate) mod granite;
pub(crate) mod llama;
pub(crate) mod mistral;
pub(crate) mod mixtral;
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

use candle_core::{Device, Result, Tensor};

use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
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
}

/// PagedAttention-specific context for transform.
pub struct PagedAttentionContext<'a> {
    /// KV cache tensors per layer: Vec<(key_cache, value_cache)>
    pub kv_cache: Vec<(Tensor, Tensor)>,

    /// PagedAttention metadata (block tables, slot mappings, etc.)
    pub metadata: &'a PagedAttentionInputMetadata,
}

/// Trait for transformer-based language models.
///
/// Extends `Model` with transformer-specific operations. Models define capabilities
/// as pure functions over tensors. Pipelines decide what to call based on their
/// stage configuration.
///
/// # Capabilities
/// - `embed`: tokens → hidden states
/// - `transform`: hidden → hidden (through layers)
/// - `lm_head`: hidden → logits
///
/// # Example: Single-node inference
/// ```ignore
/// let hidden = model.embed(&tokens)?;
/// let hidden = model.transform(hidden, &ctx)?;
/// let logits = model.lm_head(hidden)?;
/// ```
///
/// # Pipeline parallelism
/// The pipeline (not the model) determines which methods to call based on its stage:
/// - HEAD: `embed()` → `transform()` → send activation
/// - TAIL: receive → `transform()` → `lm_head()`
pub trait TransformerModel: Model {
    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize;

    /// Maximum sequence length this model supports.
    fn max_seq_len(&self) -> usize;

    /// Convert token IDs to embeddings.
    ///
    /// Input: token IDs [batch, seq_len]
    /// Output: embeddings [batch, seq_len, hidden_dim]
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;

    /// Transform hidden states through transformer layers.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: hidden states [batch, seq_len, hidden_dim]
    ///
    /// The context provides position information for RoPE and optional paged attention metadata.
    /// The cache is owned by the Pipeline and passed in - models are stateless.
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>;

    /// Project hidden states to vocabulary logits.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: logits [batch, seq_len, vocab_size]
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}
