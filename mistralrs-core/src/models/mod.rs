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

/// Trait for language models that expose their building blocks.
///
/// This trait allows pipelines to orchestrate when to call embed(), forward(), and lm_head()
/// based on pipeline stage configuration, rather than having a unified forward_inputs() that
/// always runs all operations.
///
/// The associated `State` type contains per-request state needed for forward passes:
/// - Position information (for RoPE)
/// - Paged attention metadata
/// - Any other model-specific runtime state
#[allow(dead_code)] // Intentional API surface for pipeline parallelism
pub trait LanguageModel: Send + Sync {
    /// Model-specific state for forward passes.
    /// Computed by the pipeline from Sequence state and passed to forward().
    type State;

    /// Convert token IDs to embeddings.
    ///
    /// Input: token IDs tensor [batch, seq]
    /// Output: embeddings [batch, seq, hidden_dim]
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// Forward pass through transformer layers (mutates KV cache).
    ///
    /// Input: hidden states [batch, seq, hidden_dim], input_ids tensor, per-request state
    /// Output: hidden states [batch, seq, hidden_dim]
    ///
    /// Note: input_ids parameter is needed for causal masking and KV cache position tracking.
    fn forward(&self, hidden: Tensor, input_ids: &Tensor, state: &Self::State) -> Result<Tensor>;

    /// Apply lm_head: project hidden states to vocabulary logits.
    ///
    /// Input: hidden states [batch, seq, hidden_dim]
    /// Output: logits [batch, vocab_size]
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;

    /// Check if this model contains layer 0 (first stage behavior).
    fn has_layer_0(&self) -> bool;

    /// Check if this model contains the final layer (last stage behavior).
    fn has_final_layer(&self) -> bool;
}
