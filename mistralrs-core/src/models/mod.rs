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

use candle_core::{Result, Tensor};

use crate::paged_attention::AttentionImplementation;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;

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
/// Decomposes the forward pass into three stages that can be orchestrated independently:
/// 1. `embed`: tokens → hidden states (first stage only)
/// 2. `transform`: hidden → hidden (all stages)
/// 3. `lm_head`: hidden → logits (last stage only)
///
/// This separation enables pipeline parallelism where different nodes run different stages.
/// For single-node inference, call all three in sequence.
///
/// # Example: Single-node inference
/// ```ignore
/// let hidden = model.embed(&tokens)?;
/// let hidden = model.transform(hidden, &ctx)?;
/// let logits = model.lm_head(hidden)?;
/// ```
///
/// # Example: Pipeline parallelism (HEAD stage)
/// ```ignore
/// let hidden = model.embed(&tokens)?;
/// let hidden = model.transform(hidden, &ctx)?;
/// send_to_next_stage(hidden);
/// ```
///
/// # Example: Pipeline parallelism (TAIL stage)
/// ```ignore
/// let hidden = receive_from_prev_stage();
/// let hidden = model.transform(hidden, &ctx)?;
/// let logits = model.lm_head(hidden)?;
/// ```
pub trait TransformerModel: Send + Sync {
    /// Convert token IDs to embeddings.
    ///
    /// Input: token IDs [batch, seq_len]
    /// Output: embeddings [batch, seq_len, hidden_dim]
    ///
    /// Only called by first pipeline stage (the one with token embeddings).
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;

    /// Transform hidden states through transformer layers.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: hidden states [batch, seq_len, hidden_dim]
    ///
    /// Called by all pipeline stages. Each stage processes its assigned layers.
    /// The context provides position information for RoPE and optional PagedAttention metadata.
    fn transform(&self, hidden: Tensor, ctx: &TransformContext) -> Result<Tensor>;

    /// Project hidden states to vocabulary logits.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: logits [batch, vocab_size] (typically last position extracted)
    ///
    /// Only called by last pipeline stage (the one with lm_head weights).
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;

    /// Whether this model has embedding weights (first pipeline stage).
    fn has_embed(&self) -> bool;

    /// Whether this model has lm_head weights (last pipeline stage).
    fn has_lm_head(&self) -> bool;

    /// Attention implementation used by this model.
    fn attention_impl(&self) -> AttentionImplementation;
}
