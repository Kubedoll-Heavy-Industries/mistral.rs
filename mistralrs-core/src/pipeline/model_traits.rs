//! Shared traits and types for transformer models.
//!
//! This module defines common types used across different transformer implementations.
//!
//! For the canonical transformer model trait, see [`crate::models::TokenizerModel`].

// Allow dead code - work-in-progress for unified transformer model traits
#![allow(dead_code)]

use candle_core::{DType, Device, Result, Tensor};

use crate::layers_masker::CausalMasker;

// ============================================================================
// Attention Pattern
// ============================================================================

/// Attention masking pattern - an architectural property of the model.
///
/// Determines which positions can attend to which other positions.
/// The pattern is fixed per model architecture; the actual mask tensor
/// is computed at runtime based on sequence parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionPattern {
    /// Causal (autoregressive): position i attends to positions [0..=i].
    /// Used by decoder-only models (GPT, LLaMA, Qwen, etc.)
    Causal,

    /// Bidirectional: position i attends to all positions [0..seq_len].
    /// Used by encoder models (BERT) or encoder portions of encoder-decoder.
    Bidirectional,

    /// Sliding window: position i attends to positions [i.saturating_sub(w)..=i].
    /// Used by Mistral, some Gemma variants for efficient long-context.
    SlidingWindow { window_size: usize },

    /// Chunked attention: positions attend within fixed-size chunks.
    /// Used by some efficient transformer variants.
    Chunked { chunk_size: usize },
}

impl AttentionPattern {
    /// Compute the attention mask tensor for this pattern.
    ///
    /// # Arguments
    /// * `seq_len` - Current sequence length being processed
    /// * `device` - Device to create the mask on
    /// * `past_kv_len` - Number of tokens already in KV cache
    /// * `dtype` - Data type for the mask tensor
    ///
    /// # Returns
    /// * `Some(mask)` - Mask tensor to apply during attention
    /// * `None` - No masking needed (e.g., single token decode, bidirectional)
    pub fn compute_mask(
        &self,
        seq_len: usize,
        device: &Device,
        past_kv_len: usize,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        // Single token decode never needs a mask
        if seq_len == 1 {
            return Ok(None);
        }

        match self {
            Self::Causal => CausalMasker.make_causal_mask(seq_len, device, past_kv_len, dtype),
            Self::Bidirectional => {
                // No masking for bidirectional attention
                Ok(None)
            }
            Self::SlidingWindow { window_size } => CausalMasker.make_sliding_window_mask(
                seq_len,
                device,
                past_kv_len,
                *window_size,
                dtype,
            ),
            Self::Chunked { chunk_size: _ } => {
                // TODO: Implement chunked mask properly
                // For now, fall back to causal
                CausalMasker.make_causal_mask(seq_len, device, past_kv_len, dtype)
            }
        }
    }
}
