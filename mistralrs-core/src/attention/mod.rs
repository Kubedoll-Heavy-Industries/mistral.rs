#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use crate::attention::backends::cpu;
use crate::paged_attention::PagedAttention;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::KvCache;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::QuantMethod;

mod backends;

// ============================================================================
// Attention Configuration Types
// ============================================================================

/// Configuration for attention computation - captures architectural properties.
///
/// This struct captures all the variations across transformer attention implementations:
/// - Head configuration (MHA/GQA/MQA)
/// - Q/K normalization (Qwen3-specific)
/// - Softmax modifications (softcapping for Gemma2)
/// - Sliding window attention
///
/// Position encoding (RoPE) is handled separately since it requires mutable state
/// and different models have different RoPE implementations.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for GQA; equals n_heads for MHA)
    pub n_kv_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Sliding window size (None for full attention)
    pub sliding_window: Option<usize>,
    /// Softcap for attention logits (Gemma2-specific)
    pub softcap: Option<f32>,
    /// Custom softmax scale (defaults to 1/√head_dim if None)
    pub softmax_scale: Option<f32>,
}

impl AttentionConfig {
    /// Create a new attention configuration.
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            n_heads,
            n_kv_heads,
            head_dim,
            sliding_window: None,
            softcap: None,
            softmax_scale: None,
        }
    }

    /// Set sliding window size.
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Set softcap for attention logits (Gemma2).
    pub fn with_softcap(mut self, cap: f32) -> Self {
        self.softcap = Some(cap);
        self
    }

    /// Set custom softmax scale.
    pub fn with_softmax_scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Number of KV groups (n_heads / n_kv_heads).
    #[inline]
    pub fn n_kv_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Effective softmax scale (custom or 1/√head_dim).
    #[inline]
    pub fn effective_softmax_scale(&self) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }

    /// Convert to SdpaParams for the SDPA kernel.
    pub fn to_sdpa_params(&self) -> SdpaParams {
        SdpaParams {
            n_kv_groups: self.n_kv_groups(),
            softcap: self.softcap,
            softmax_scale: self.effective_softmax_scale(),
            sliding_window: self.sliding_window,
        }
    }
}

/// Q/K normalization configuration (Qwen3-specific).
///
/// Some models apply per-head RMSNorm to Q and K after projection
/// but before RoPE. This is separate from attention config because
/// it requires weight tensors.
pub trait QkNorm: Send + Sync {
    /// Apply normalization to Q tensor.
    /// Input: [batch, n_heads, seq_len, head_dim] flattened to [batch * n_heads * seq_len, head_dim]
    fn forward_q(&self, q: &Tensor) -> Result<Tensor>;
    /// Apply normalization to K tensor.
    fn forward_k(&self, k: &Tensor) -> Result<Tensor>;
}

/// Position encoding trait - abstracts over RoPE variants.
///
/// Different models use different position encoding schemes:
/// - Standard RoPE (LLaMA, Mistral)
/// - Partial RoPE (Phi2/3 - only applies to portion of head_dim)
/// - YaRN extended context (Mistral3)
pub trait PositionEncoding: Send + Sync {
    /// Apply position encoding to Q and K tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, n_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, n_kv_heads, seq_len, head_dim]
    /// * `seqlen_offsets` - Position offset for each sequence in batch
    ///
    /// # Returns
    /// (q_with_pos, k_with_pos) with same shapes as input
    fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offsets: &[usize]) -> Result<(Tensor, Tensor)>;
}

/// Attention mechanism trait - abstracts over attention implementations.
///
/// This trait enables composable transformer blocks by allowing different
/// attention implementations:
/// - `CausalAttention` (generic, handles Q/K norm, RoPE, paged attention)
/// - Model-specific inline attention (for custom optimizations)
pub trait Attention: Send + Sync {
    /// Forward pass through the attention layer.
    ///
    /// # Arguments
    /// * `x` - Input hidden states after pre-attention norm [batch, seq_len, hidden_dim]
    /// * `mask` - Optional attention mask
    /// * `cache` - KV cache (mutated for eager attention)
    /// * `position_offsets` - Position offsets for RoPE
    /// * `metadata` - Paged attention metadata (KV cache tensors + input metadata)
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_dim]
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        position_offsets: &[usize],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor>;

    /// Check if this attention uses paged attention.
    fn has_paged_attn(&self) -> bool;
}

// ============================================================================
// Concrete QkNorm Implementations
// ============================================================================

/// Paired RMSNorm for Q and K tensors (Qwen3-style).
///
/// Qwen3 applies per-head RMSNorm to Q and K after projection but before RoPE.
/// The input tensors should be flattened to [batch * n_heads * seq_len, head_dim].
pub struct RmsNormQkNorm<N> {
    q_norm: N,
    k_norm: N,
}

impl<N> RmsNormQkNorm<N> {
    pub fn new(q_norm: N, k_norm: N) -> Self {
        Self { q_norm, k_norm }
    }
}

/// Implement QkNorm for any norm type that implements Module (candle_nn::Module trait).
impl<N: candle_nn::Module + Send + Sync> QkNorm for RmsNormQkNorm<N> {
    fn forward_q(&self, q: &Tensor) -> Result<Tensor> {
        self.q_norm.forward(q)
    }

    fn forward_k(&self, k: &Tensor) -> Result<Tensor> {
        self.k_norm.forward(k)
    }
}

// ============================================================================
// Generic Causal Attention
// ============================================================================

/// Generic causal attention module that handles all architectural variations.
///
/// This struct consolidates the duplicated attention logic across 8+ model
/// implementations into a single, configurable component.
///
/// # Architecture Support
/// - MHA, GQA, MQA (via n_heads/n_kv_heads configuration)
/// - Optional Q/K normalization (Qwen3)
/// - Optional sliding window attention (Mistral, Phi3)
/// - Optional softcapping (Gemma2)
/// - Eager or paged KV cache
/// - Optional dtype conversion for quantized models
///
/// # Example
/// ```ignore
/// let attn = CausalAttention::new(config, q_proj, k_proj, v_proj, o_proj, rope);
/// let output = attn.forward(&hidden, mask, &mut cache, seqlen_offsets, paged_attn)?;
/// ```
pub struct CausalAttention {
    /// Attention configuration (head counts, sliding window, etc.)
    config: AttentionConfig,
    /// Query projection
    q_proj: Arc<dyn QuantMethod>,
    /// Key projection
    k_proj: Arc<dyn QuantMethod>,
    /// Value projection
    v_proj: Arc<dyn QuantMethod>,
    /// Output projection
    o_proj: Arc<dyn QuantMethod>,
    /// Position encoding (RoPE)
    rope: Arc<dyn PositionEncoding>,
    /// Optional Q/K normalization (Qwen3)
    qk_norm: Option<Arc<dyn QkNorm>>,
    /// PagedAttention module (if using paged attention)
    paged_attn: Option<PagedAttention>,
    /// Target dtype for QKV after RoPE (for quantized models)
    attn_dtype: Option<DType>,
}

impl CausalAttention {
    /// Create a new causal attention module.
    pub fn new(
        config: AttentionConfig,
        q_proj: Arc<dyn QuantMethod>,
        k_proj: Arc<dyn QuantMethod>,
        v_proj: Arc<dyn QuantMethod>,
        o_proj: Arc<dyn QuantMethod>,
        rope: Arc<dyn PositionEncoding>,
    ) -> Self {
        Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            qk_norm: None,
            paged_attn: None,
            attn_dtype: None,
        }
    }

    /// Add Q/K normalization (for Qwen3).
    pub fn with_qk_norm(mut self, qk_norm: Arc<dyn QkNorm>) -> Self {
        self.qk_norm = Some(qk_norm);
        self
    }

    /// Add paged attention support.
    pub fn with_paged_attn(mut self, paged_attn: PagedAttention) -> Self {
        self.paged_attn = Some(paged_attn);
        self
    }

    /// Set target dtype for attention computation (for quantized models).
    /// QKV tensors will be converted to this dtype after RoPE and before SDPA.
    pub fn with_attn_dtype(mut self, dtype: DType) -> Self {
        self.attn_dtype = Some(dtype);
        self
    }

    /// Check if this attention uses paged attention.
    pub fn has_paged_attn(&self) -> bool {
        self.paged_attn.is_some()
    }

    // =========================================================================
    // Mutable Accessors (for ISQ quantization)
    // =========================================================================

    /// Get mutable references to all projection weights (for ISQ).
    /// Returns (q_proj, k_proj, v_proj, o_proj).
    #[allow(clippy::type_complexity)]
    pub fn projections_mut(
        &mut self,
    ) -> (
        &mut Arc<dyn QuantMethod>,
        &mut Arc<dyn QuantMethod>,
        &mut Arc<dyn QuantMethod>,
        &mut Arc<dyn QuantMethod>,
    ) {
        (
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        )
    }

    /// Forward pass through the attention layer.
    ///
    /// # Arguments
    /// * `hidden` - Input hidden states [batch, seq_len, hidden_dim]
    /// * `mask` - Optional attention mask
    /// * `cache` - KV cache (mutated for eager attention)
    /// * `seqlen_offsets` - Position offsets for RoPE
    /// * `paged_metadata` - Paged attention metadata (if using paged attention)
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        seqlen_offsets: &[usize],
        paged_metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden.dims3()?;
        let cfg = &self.config;

        // 1. QKV projections
        let q = self.q_proj.forward(hidden)?;
        let k = self.k_proj.forward(hidden)?;
        let v = self.v_proj.forward(hidden)?;

        // 2. Reshape for multi-head attention
        let mut q = reshape_for_attn(q, b_sz, seq_len, cfg.n_heads, cfg.head_dim)?;
        let mut k = reshape_for_attn(k, b_sz, seq_len, cfg.n_kv_heads, cfg.head_dim)?;
        let v = reshape_for_attn(v, b_sz, seq_len, cfg.n_kv_heads, cfg.head_dim)?;

        // 3. Optional Q/K normalization (Qwen3)
        if let Some(ref qk_norm) = self.qk_norm {
            // Flatten for per-head norm: [b, h, s, d] -> [b*h*s, d]
            let q_flat = q.flatten(0, 2)?;
            let k_flat = k.flatten(0, 2)?;
            let q_normed = qk_norm.forward_q(&q_flat)?;
            let k_normed = qk_norm.forward_k(&k_flat)?;
            q = q_normed.reshape((b_sz, cfg.n_heads, seq_len, cfg.head_dim))?;
            k = k_normed.reshape((b_sz, cfg.n_kv_heads, seq_len, cfg.head_dim))?;
        }

        // 4. Apply position encoding (RoPE)
        let (q, k) = self.rope.forward(&q, &k, seqlen_offsets)?;

        // 5. Optional dtype conversion (for quantized models)
        let (q, k, v) = if let Some(dtype) = self.attn_dtype {
            (q.to_dtype(dtype)?, k.to_dtype(dtype)?, v.to_dtype(dtype)?)
        } else {
            (q, k, v)
        };

        // 6. Scaled dot-product attention with cache
        let sdpa_params = cfg.to_sdpa_params();
        let y = sdpa_with_cache(
            &q,
            &k,
            &v,
            mask,
            cache,
            self.paged_attn.as_ref(),
            paged_metadata,
            &sdpa_params,
        )?;

        // 7. Reshape output and apply output projection
        let y = reshape_attn_output(y, b_sz, seq_len, mask.is_some())?;
        // Convert back to input dtype if we changed it for attention
        let y = if self.attn_dtype.is_some() {
            y.to_dtype(hidden.dtype())?
        } else {
            y
        };
        self.o_proj.forward(&y)
    }
}

impl Attention for CausalAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        position_offsets: &[usize],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        CausalAttention::forward(self, x, mask, cache, position_offsets, metadata)
    }

    fn has_paged_attn(&self) -> bool {
        CausalAttention::has_paged_attn(self)
    }
}

// ============================================================================
// Fused QKV Causal Attention (Phi3-style)
// ============================================================================

/// Causal attention with fused QKV projection (Phi3-style).
///
/// Unlike `CausalAttention` which has separate Q, K, V projections, this variant
/// uses a single fused `qkv_proj` that outputs Q, K, V concatenated. The output
/// is split in the forward pass.
///
/// Weight layout: `[hidden_size, n_heads * head_dim + 2 * n_kv_heads * head_dim]`
/// Output order: Q (n_heads * head_dim), K (n_kv_heads * head_dim), V (n_kv_heads * head_dim)
///
/// # Example
/// ```ignore
/// let attn = FusedQkvCausalAttention::new(config, qkv_proj, o_proj, rope);
/// let output = attn.forward(&hidden, mask, &mut cache, seqlen_offsets, paged_attn)?;
/// ```
pub struct FusedQkvCausalAttention {
    /// Attention configuration (head counts, sliding window, etc.)
    config: AttentionConfig,
    /// Fused Q+K+V projection
    qkv_proj: Arc<dyn QuantMethod>,
    /// Output projection
    o_proj: Arc<dyn QuantMethod>,
    /// Position encoding (RoPE)
    rope: Arc<dyn PositionEncoding>,
    /// Optional Q/K normalization
    qk_norm: Option<Arc<dyn QkNorm>>,
    /// PagedAttention module (if using paged attention)
    paged_attn: Option<PagedAttention>,
    /// Target dtype for QKV after RoPE (for quantized models)
    attn_dtype: Option<DType>,
}

impl FusedQkvCausalAttention {
    /// Create a new fused QKV causal attention module.
    pub fn new(
        config: AttentionConfig,
        qkv_proj: Arc<dyn QuantMethod>,
        o_proj: Arc<dyn QuantMethod>,
        rope: Arc<dyn PositionEncoding>,
    ) -> Self {
        Self {
            config,
            qkv_proj,
            o_proj,
            rope,
            qk_norm: None,
            paged_attn: None,
            attn_dtype: None,
        }
    }

    /// Add Q/K normalization.
    pub fn with_qk_norm(mut self, qk_norm: Arc<dyn QkNorm>) -> Self {
        self.qk_norm = Some(qk_norm);
        self
    }

    /// Add paged attention support.
    pub fn with_paged_attn(mut self, paged_attn: PagedAttention) -> Self {
        self.paged_attn = Some(paged_attn);
        self
    }

    /// Set target dtype for attention computation (for quantized models).
    pub fn with_attn_dtype(mut self, dtype: DType) -> Self {
        self.attn_dtype = Some(dtype);
        self
    }

    /// Check if this attention uses paged attention.
    pub fn has_paged_attn(&self) -> bool {
        self.paged_attn.is_some()
    }

    /// Forward pass through the attention layer.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        seqlen_offsets: &[usize],
        paged_metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden.dims3()?;
        let cfg = &self.config;

        // 1. Fused QKV projection
        let qkv = self.qkv_proj.forward(hidden)?;

        // 2. Split QKV output
        // Layout: [batch, seq_len, n_heads * head_dim + 2 * n_kv_heads * head_dim]
        let q_size = cfg.n_heads * cfg.head_dim;
        let kv_size = cfg.n_kv_heads * cfg.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // 3. Reshape for multi-head attention
        let mut q = reshape_for_attn(q, b_sz, seq_len, cfg.n_heads, cfg.head_dim)?;
        let mut k = reshape_for_attn(k, b_sz, seq_len, cfg.n_kv_heads, cfg.head_dim)?;
        let v = reshape_for_attn(v, b_sz, seq_len, cfg.n_kv_heads, cfg.head_dim)?;

        // 4. Optional Q/K normalization
        if let Some(ref qk_norm) = self.qk_norm {
            let q_flat = q.flatten(0, 2)?;
            let k_flat = k.flatten(0, 2)?;
            let q_normed = qk_norm.forward_q(&q_flat)?;
            let k_normed = qk_norm.forward_k(&k_flat)?;
            q = q_normed.reshape((b_sz, cfg.n_heads, seq_len, cfg.head_dim))?;
            k = k_normed.reshape((b_sz, cfg.n_kv_heads, seq_len, cfg.head_dim))?;
        }

        // 5. Apply position encoding (RoPE)
        let (q, k) = self.rope.forward(&q, &k, seqlen_offsets)?;

        // 6. Optional dtype conversion (for quantized models)
        let (q, k, v) = if let Some(dtype) = self.attn_dtype {
            (q.to_dtype(dtype)?, k.to_dtype(dtype)?, v.to_dtype(dtype)?)
        } else {
            (q, k, v)
        };

        // 7. Scaled dot-product attention with cache
        let sdpa_params = cfg.to_sdpa_params();
        let y = sdpa_with_cache(
            &q,
            &k,
            &v,
            mask,
            cache,
            self.paged_attn.as_ref(),
            paged_metadata,
            &sdpa_params,
        )?;

        // 8. Reshape output and apply output projection
        let y = reshape_attn_output(y, b_sz, seq_len, mask.is_some())?;
        let y = if self.attn_dtype.is_some() {
            y.to_dtype(hidden.dtype())?
        } else {
            y
        };
        self.o_proj.forward(&y)
    }
}

impl Attention for FusedQkvCausalAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        position_offsets: &[usize],
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        FusedQkvCausalAttention::forward(self, x, mask, cache, position_offsets, metadata)
    }

    fn has_paged_attn(&self) -> bool {
        FusedQkvCausalAttention::has_paged_attn(self)
    }
}

#[allow(unused)]
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        // For short sequences, use the regular path
        return attention_fn(q, k, v, mask);
    }

    // Chunk the query to avoid OOM on long sequences
    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
                }
            })
            .transpose()?;

        // Compute attention for this chunk
        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref())?;

        attn_chunks.push(att_chunk);
    }

    // Concatenate all chunks along the sequence dimension
    Tensor::cat(&attn_chunks, 2)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// The attention implementation is dispatched as follows:
    /// 1) If using flash attn (CUDA), use a flash attention V2/V3 kernel
    /// 2) If decoding and using a Metal device, use a fused kkernel
    /// 2) Otherwise, use the "naive" SDPA implementation (with optimized mask+softmax+scale application)
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda() && crate::using_flash_attn() && q.dtype() != DType::F32;

        if can_use_flash {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                match q.dtype() {
                    DType::F32 => {
                        return cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params);
                    }
                    DType::F16 => {
                        return cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params)
                    }
                    DType::BF16 => {
                        return cpu::run_flash_attn_cpu::<half::bf16>(
                            &q,
                            &k,
                            &v,
                            mask,
                            sdpa_params,
                        );
                    }
                    _ => {
                        return Err(candle_core::Error::Msg("Unsupported data type".into()));
                    }
                }
            } else {
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        self.run_attention_noflash(q, k, v, mask, sdpa_params)
    }

    /// Same as `run_attention`, but no flash attention
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = if seq_len == 1 {
            &[32, 64, 72, 80, 96, 128, 256]
        } else {
            // Not sure why the full kernel doesn't like 256.
            // [32, 64, 72, 80, 96, 128, 256]
            &[32, 64, 72, 80, 96, 128]
        };
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            return candle_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                false,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || mistralrs_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
            );
        }

        // TODO: bench?
        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                chunked_attention(q, &k, &v, mask, |q_chunk, _k, _v, mask_chunk| {
                    // cuBLASLt batch matmul implementation requires inputs to be dims3
                    let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                        q_chunk.dims4()?;
                    let q_flat = q_chunk.flatten(0, 1)?;

                    let attention_bias = match mask_chunk {
                        Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                            Some(mask.repeat((chunk_n_heads, 1, 1))?)
                        }
                        Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                        Some(mask) if mask.rank() == 4 => {
                            let tgt_shape =
                                vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                            Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                        }
                        Some(mask) => {
                            candle_core::bail!("cublaslt attn mask: rank must be 3 or 4")
                        }
                        None => None,
                    };

                    // If attention_bias is set, we fuse the add by giving it as the output matrix
                    // and setting beta to 1.0
                    let beta = match attention_bias.is_some() {
                        true => Some(1.0),
                        false => None,
                    };

                    // Batch matrix multiplication
                    // Fuse softmax scale and attention_bias add
                    let mut attention_scores = cublaslt.batch_matmul(
                        &k_flat,
                        &q_flat,
                        attention_bias.as_ref(),
                        Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                        beta,
                        None,
                        None,
                    )?;
                    if let Some(softcap) = sdpa_params.softcap {
                        attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                    }
                    attention_scores = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                    let context_layer = cublaslt.batch_matmul(
                        &v_flat.t()?.contiguous()?,
                        &attention_scores,
                        // We save one allocation
                        Some(&q_flat),
                        None,
                        None,
                        None,
                        None,
                    )?;

                    // Reshape to dims4
                    context_layer.reshape((chunk_b_sz, chunk_n_heads, chunk_seq_len, v_head_dim))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}

// ============================================================================
// Attention Helpers
// ============================================================================

/// Reshape a tensor for multi-head attention.
///
/// Converts from `[batch, seq_len, n_heads * head_dim]` to `[batch, n_heads, seq_len, head_dim]`.
/// Uses an optimized path for single-token decode (seq_len == 1) that avoids transpose.
///
/// # Arguments
/// * `x` - Input tensor of shape `[batch, seq_len, n_heads * head_dim]`
/// * `b_sz` - Batch size
/// * `seq_len` - Sequence length
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
pub fn reshape_for_attn(
    x: Tensor,
    b_sz: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if seq_len != 1 {
        x.reshape((b_sz, seq_len, n_heads, head_dim))?.transpose(1, 2)
    } else {
        x.reshape((b_sz, n_heads, seq_len, head_dim))
    }
}

/// Reshape attention output back to `[batch, seq_len, hidden_dim]`.
///
/// Handles the difference between prefill (needs transpose) and decode (no transpose needed).
///
/// # Arguments
/// * `y` - Attention output of shape `[batch, n_heads, seq_len, head_dim]`
/// * `b_sz` - Batch size
/// * `seq_len` - Sequence length
/// * `had_mask` - Whether a mask was used (indicates prefill vs decode)
pub fn reshape_attn_output(y: Tensor, b_sz: usize, seq_len: usize, had_mask: bool) -> Result<Tensor> {
    if had_mask {
        y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))
    } else {
        y.reshape((b_sz, seq_len, ()))
    }
}

/// Run scaled dot-product attention with KV cache support.
///
/// Dispatches to either paged attention or eager attention based on configuration.
/// For eager attention, appends K/V to the cache before computing attention.
///
/// # Arguments
/// * `q` - Query tensor `[batch, n_heads, seq_len, head_dim]`
/// * `k` - Key tensor `[batch, n_kv_heads, seq_len, head_dim]`
/// * `v` - Value tensor `[batch, n_kv_heads, seq_len, head_dim]`
/// * `mask` - Optional attention mask
/// * `kv_cache` - KV cache for eager attention (mutated to append new K/V)
/// * `paged_attn` - PagedAttention module if using paged attention
/// * `paged_metadata` - Paged attention metadata (KV cache tensors + input metadata)
/// * `sdpa_params` - SDPA parameters (softmax scale, softcap, etc.)
#[allow(clippy::too_many_arguments)]
pub fn sdpa_with_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    kv_cache: &mut KvCache,
    paged_attn: Option<&PagedAttention>,
    paged_metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    match paged_attn {
        Some(pa) => {
            let ((key_cache, value_cache), input_metadata) = paged_metadata
                .expect("paged_metadata required when using PagedAttention");
            pa.forward(
                q,
                k,
                v,
                mask,
                Some(key_cache),
                Some(value_cache),
                input_metadata,
                sdpa_params,
                None,
            )
        }
        None => {
            let (k, v) = kv_cache.append(k, v)?;
            Sdpa.run_attention(q, &k, &v, mask, None, sdpa_params)
        }
    }
}
