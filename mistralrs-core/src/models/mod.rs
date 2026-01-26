// Allow deprecated traits during migration to new trait hierarchy
#![allow(deprecated)]

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

/// Trait for transformer-based models.
///
/// Extends `Model` with core transformer operations: embedding and layer-by-layer
/// transformation. This trait is shared by all transformer architectures (language
/// models, embedding models, vision encoders, etc.).
///
/// # Capabilities
/// - `embed`: tokens → hidden states
/// - `transform`: hidden → hidden (through layers)
///
/// # Example: Embedding model
/// ```ignore
/// let hidden = model.embed(&tokens)?;
/// let hidden = model.transform(hidden, &ctx)?;
/// // Apply pooling for embeddings
/// ```
///
/// # Pipeline parallelism
/// The pipeline (not the model) determines which methods to call based on its stage.
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
}

/// Trait for language models (decoder-only transformers for text generation).
///
/// Extends `TransformerModel` with the language modeling head (`lm_head`) that
/// projects hidden states to vocabulary logits for next-token prediction.
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
pub trait LanguageModel: TransformerModel {
    /// Project hidden states to vocabulary logits.
    ///
    /// Input: hidden states [batch, seq_len, hidden_dim]
    /// Output: logits [batch, seq_len, vocab_size]
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}

// =============================================================================
// Unified Model Configuration Traits
// =============================================================================
//
// These traits abstract over configuration sources (JSON config.json vs GGUF
// metadata), enabling a single model implementation to be constructed from
// either source.

use crate::layers::Activation;

/// Configuration trait for Llama-family models.
///
/// Implemented by both JSON config (safetensors) and GGUF metadata, enabling
/// a single `Llama` model to be constructed from either source.
///
/// # Example
/// ```ignore
/// // Works with JSON config
/// let model = Llama::from_config(&json_config, vb)?;
///
/// // Works with GGUF metadata
/// let model = Llama::from_config(&gguf_config, content)?;
/// ```
pub trait LlamaConfig {
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
}
