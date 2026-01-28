//! Type-safe routing strategies for Mixture of Experts.
//!
//! This module provides a trait-based abstraction for MoE routing, enabling
//! type-safe dispatch at compile time while supporting multiple routing
//! strategies through the same interface.
//!
//! # Design Rationale
//!
//! Instead of runtime enum dispatch for routing strategies (greedy vs group-limited),
//! we use a trait with marker types. This enables:
//! - Zero-cost abstractions via monomorphization
//! - Type-safe composition with `MoE<R: RoutingStrategy>`
//! - Invalid routing configurations become compile-time errors
//!
//! # Supported Strategies
//!
//! - [`SoftmaxTopK`]: Standard softmax top-k routing (Mixtral, Qwen3 MoE)
//! - [`GroupLimitedGreedy`]: Group-based selection with per-group limits (DeepSeek V2)

use candle_core::{Result, Tensor, D};
use candle_nn::ops::softmax_last_dim;

// ============================================================================
// Core Types
// ============================================================================

/// Output from routing computation.
///
/// Contains the routing weights and expert indices for each token position.
/// Both tensors have shape `[batch * seq_len, top_k]`.
#[derive(Debug)]
pub struct RouteOutput {
    /// Routing weights (normalized or scaled). Shape: `[batch * seq_len, top_k]`
    pub weights: Tensor,
    /// Expert indices for each token. Shape: `[batch * seq_len, top_k]`
    pub indices: Tensor,
}

/// Configuration for routing.
///
/// This contains parameters common to all routing strategies.
/// Strategy-specific parameters should be stored in the strategy struct itself.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Total number of experts in the model
    pub num_experts: usize,
    /// Number of experts to route each token to
    pub top_k: usize,
    /// Whether to normalize top-k weights to sum to 1
    pub normalize_weights: bool,
    /// Scaling factor for routing weights (used when not normalizing)
    pub scaling_factor: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            normalize_weights: true,
            scaling_factor: 1.0,
        }
    }
}

// ============================================================================
// Routing Strategy Trait
// ============================================================================

/// Trait for MoE routing strategies.
///
/// Routing strategies determine how tokens are assigned to experts.
/// This trait enables type-level dispatch through `MoE<R: RoutingStrategy>`,
/// eliminating runtime overhead from routing strategy selection.
///
/// # Implementation Guidelines
///
/// - `route()` should NOT modify the gate projection - it receives raw logits
/// - Normalization is controlled by `RoutingConfig::normalize_weights`
/// - Return indices should be contiguous for efficient expert dispatch
///
/// # Example
///
/// ```ignore
/// struct MoE<R: RoutingStrategy> {
///     gate: Linear,
///     experts: MoEExperts,
///     config: RoutingConfig,
///     _routing: PhantomData<R>,
/// }
///
/// impl<R: RoutingStrategy> FeedForward for MoE<R> {
///     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
///         let logits = self.gate.forward(xs)?;
///         let RouteOutput { weights, indices } = R::route(&logits, &self.config)?;
///         self.experts.forward(xs, weights, &indices)
///     }
/// }
/// ```
pub trait RoutingStrategy: Send + Sync + 'static {
    /// Compute routing weights and expert indices from router logits.
    ///
    /// # Arguments
    /// * `logits` - Router logits of shape `[batch * seq_len, num_experts]`
    /// * `config` - Routing configuration
    ///
    /// # Returns
    /// `RouteOutput` containing weights and indices, both shape `[batch * seq_len, top_k]`
    fn route(logits: &Tensor, config: &RoutingConfig) -> Result<RouteOutput>;
}

// ============================================================================
// Softmax Top-K Routing
// ============================================================================

/// Standard softmax top-k routing.
///
/// This is the most common routing strategy, used by:
/// - Mixtral
/// - Qwen3 MoE
/// - Standard sparse MoE implementations
///
/// # Algorithm
///
/// 1. Apply softmax to router logits to get probabilities
/// 2. Select top-k experts by probability
/// 3. Optionally normalize selected weights to sum to 1
///
/// # Type Parameter
///
/// This struct uses no type parameters - the configuration is passed at runtime.
/// For compile-time top-k specification, see `SoftmaxTopKConst<const K: usize>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftmaxTopK;

impl RoutingStrategy for SoftmaxTopK {
    fn route(logits: &Tensor, config: &RoutingConfig) -> Result<RouteOutput> {
        // 1. Softmax over experts dimension
        let routing_weights = softmax_last_dim(&logits.to_dtype(candle_core::DType::F32)?)?;

        // 2. Get top-k indices by sorting
        let topk_indices = routing_weights
            .arg_sort_last_dim(false)? // descending
            .narrow(D::Minus1, 0, config.top_k)?
            .contiguous()?;

        // 3. Gather top-k weights
        let mut topk_weights = routing_weights.gather(&topk_indices, D::Minus1)?;

        // 4. Normalize or scale
        if config.normalize_weights {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        } else if (config.scaling_factor - 1.0).abs() > f64::EPSILON {
            topk_weights = (topk_weights * config.scaling_factor)?;
        }

        Ok(RouteOutput {
            weights: topk_weights,
            indices: topk_indices,
        })
    }
}

// ============================================================================
// Group-Limited Greedy Routing
// ============================================================================

/// Group-limited greedy routing (DeepSeek V2 style).
///
/// Experts are divided into groups, and routing is limited to selecting
/// top-k groups first, then top-k experts within those groups.
///
/// # Algorithm
///
/// 1. Compute scores for each expert (softmax of logits)
/// 2. Group experts and compute max score per group
/// 3. Select top-k groups
/// 4. Mask out experts not in selected groups
/// 5. Select top-k experts from remaining
///
/// # Use Cases
///
/// - DeepSeek V2 with grouped experts
/// - Models requiring balanced routing across expert groups
#[derive(Debug, Clone)]
pub struct GroupLimitedGreedy {
    /// Number of expert groups
    pub n_groups: usize,
    /// Number of top groups to select
    pub topk_groups: usize,
}

impl GroupLimitedGreedy {
    /// Create a new GroupLimitedGreedy routing strategy.
    ///
    /// # Arguments
    /// * `n_groups` - Number of expert groups (must divide num_experts evenly)
    /// * `topk_groups` - Number of top groups to select (must be <= n_groups)
    pub fn new(n_groups: usize, topk_groups: usize) -> Self {
        Self {
            n_groups,
            topk_groups,
        }
    }

    /// Route tokens using group-limited greedy strategy.
    ///
    /// This is similar to the trait method but requires `&self` to access
    /// group configuration. For trait-based routing, use `route_with_config`.
    pub fn route_with_config(
        &self,
        logits: &Tensor,
        config: &RoutingConfig,
    ) -> Result<RouteOutput> {
        let (batch_seq_len, num_experts) = logits.dims2()?;
        let experts_per_group = num_experts / self.n_groups;

        // 1. Compute routing weights via softmax
        let scores = softmax_last_dim(&logits.to_dtype(candle_core::DType::F32)?)?;

        // 2. Compute max score per group: reshape to (n, n_group, experts_per_group), then max
        let group_scores = scores
            .reshape((batch_seq_len, self.n_groups, experts_per_group))?
            .max(D::Minus1)?;

        // 3. Select top-k groups
        let group_indices = group_scores
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.topk_groups)?
            .contiguous()?;

        // 4. Create group mask: 1 for selected groups, 0 otherwise
        let mut group_mask = group_scores.zeros_like()?;
        let ones_for_scatter = group_indices.ones_like()?.to_dtype(group_mask.dtype())?;
        group_mask = group_mask.scatter_add(&group_indices, &ones_for_scatter, 1)?;

        // 5. Expand group mask to expert-level mask
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((batch_seq_len, self.n_groups, experts_per_group))?
            .reshape((batch_seq_len, num_experts))?;

        // 6. Mask out experts not in selected groups
        // Invert mask: where mask == 0, set score to 0
        let masked_scores = scores.broadcast_mul(&score_mask)?;

        // 7. Select top-k from masked scores
        let topk_indices = masked_scores
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, config.top_k)?
            .contiguous()?;

        let mut topk_weights = scores.gather(&topk_indices, D::Minus1)?;

        // 8. Normalize or scale
        if config.normalize_weights {
            let denominator = (topk_weights.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weights = topk_weights.broadcast_div(&denominator)?;
        } else if (config.scaling_factor - 1.0).abs() > f64::EPSILON {
            topk_weights = (topk_weights * config.scaling_factor)?;
        }

        Ok(RouteOutput {
            weights: topk_weights,
            indices: topk_indices,
        })
    }
}

// Note: GroupLimitedGreedy cannot implement RoutingStrategy directly because
// it requires per-instance configuration (n_groups, topk_groups). Instead,
// use the MoEWithGroupRouting struct which embeds the GroupLimitedGreedy config.

// ============================================================================
// SparseMixer Routing (Phi-3.5 MoE style)
// ============================================================================

/// SparseMixer routing strategy (Phi-3.5 MoE style).
///
/// This routing strategy uses jittered threshold-based masking to select
/// the top-2 experts. It differs from standard softmax top-k in several ways:
///
/// 1. **Argmax selection**: Uses argmax to find the top expert, not sorting
/// 2. **Threshold masking**: Creates a sparse mask based on score differences
/// 3. **Sequential selection**: Masks out top-1 before finding top-2
/// 4. **Per-expert softmax**: Applies softmax after masking, not before
///
/// # Algorithm
///
/// For each token:
/// 1. Find top-1 expert via argmax
/// 2. Create threshold mask: `(top1_score - score) / min(|score|, top1_score) > 2*jitter_eps`
/// 3. Apply mask, then softmax to get top-1 weight
/// 4. Mask out top-1 expert completely
/// 5. Repeat steps 1-3 on masked scores to get top-2 expert and weight
/// 6. Return both experts and their weights
///
/// # Use Cases
///
/// - Phi-3.5 MoE models
/// - Models requiring jitter-based sparsification
#[derive(Debug, Clone)]
pub struct SparseMixer {
    /// Jitter noise epsilon for threshold computation
    pub jitter_eps: f64,
}

impl SparseMixer {
    /// Create a new SparseMixer routing strategy.
    ///
    /// # Arguments
    /// * `jitter_eps` - Jitter epsilon for threshold computation (typically small, e.g., 0.01)
    pub fn new(jitter_eps: f64) -> Self {
        Self { jitter_eps }
    }

    /// Route tokens using SparseMixer strategy.
    ///
    /// This always selects exactly 2 experts per token (top_k in config is ignored).
    ///
    /// # Arguments
    /// * `logits` - Router logits of shape `[batch * seq_len, num_experts]`
    /// * `_config` - Routing configuration (top_k is ignored, always 2)
    ///
    /// # Returns
    /// `RouteOutput` with weights and indices, both shape `[batch * seq_len, 2]`
    pub fn route_with_config(
        &self,
        logits: &Tensor,
        _config: &RoutingConfig,
    ) -> Result<RouteOutput> {
        let scores = logits.to_dtype(candle_core::DType::F32)?;

        // === Top-1 Selection ===

        // 1. Find top-1 expert via argmax
        let top1_idx = scores.argmax_keepdim(D::Minus1)?;
        let top1_score = scores.gather(&top1_idx, D::Minus1)?;

        // 2. Create threshold mask
        // mask = (top1_score - score) / min(|score|, top1_score) > 2*jitter_eps
        let factor = scores.abs()?.broadcast_minimum(&top1_score)?;
        let score_diff = top1_score.broadcast_sub(&scores)?;
        let threshold_ratio = score_diff.broadcast_div(&(factor + 1e-10)?)?;
        let mask1 = threshold_ratio.gt(2.0 * self.jitter_eps)?;

        // 3. Apply mask (set masked positions to -inf) and softmax
        let masked_scores1 = masked_fill(&scores, &mask1, f64::NEG_INFINITY)?;
        let softmax_scores1 = softmax_last_dim(&masked_scores1)?;
        let top1_weight = softmax_scores1.gather(&top1_idx, D::Minus1)?;

        // === Top-2 Selection ===

        // 4. Mask out the top-1 expert completely
        let neg_inf_tensor = (scores.ones_like()? * f64::NEG_INFINITY)?;
        let scores_without_top1 = scores.scatter_add(
            &top1_idx.broadcast_as(scores.shape())?.contiguous()?,
            &neg_inf_tensor,
            D::Minus1,
        )?;

        // 5. Find top-2 expert via argmax on masked scores
        let top2_idx = scores_without_top1.argmax_keepdim(D::Minus1)?;
        let top2_score = scores_without_top1.gather(&top2_idx, D::Minus1)?;

        // 6. Create threshold mask for top-2
        let factor2 = scores.abs()?.broadcast_minimum(&top2_score)?;
        let score_diff2 = top2_score.broadcast_sub(&scores)?;
        let threshold_ratio2 = score_diff2.broadcast_div(&(factor2 + 1e-10)?)?;
        let mask2 = threshold_ratio2.gt(2.0 * self.jitter_eps)?;

        // 7. Apply mask and softmax for top-2
        let masked_scores2 = masked_fill(&scores_without_top1, &mask2, f64::NEG_INFINITY)?;
        let softmax_scores2 = softmax_last_dim(&masked_scores2)?;
        let top2_weight = softmax_scores2.gather(&top2_idx, D::Minus1)?;

        // === Combine Results ===

        // Concatenate: [top1, top2] for each token
        let weights = Tensor::cat(&[top1_weight, top2_weight], D::Minus1)?;
        let indices = Tensor::cat(&[top1_idx, top2_idx], D::Minus1)?.contiguous()?;

        Ok(RouteOutput { weights, indices })
    }
}

/// Helper function to fill tensor positions where mask is true with a value.
fn masked_fill(tensor: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor> {
    let value_tensor = (tensor.ones_like()? * value)?;
    mask.where_cond(&value_tensor, tensor)
}

// ============================================================================
// Helper for creating routing configs
// ============================================================================

impl RoutingConfig {
    /// Create a new routing config for standard normalized top-k routing.
    pub fn new_normalized(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            normalize_weights: true,
            scaling_factor: 1.0,
        }
    }

    /// Create a new routing config with scaling factor (no normalization).
    pub fn new_scaled(num_experts: usize, top_k: usize, scaling_factor: f64) -> Self {
        Self {
            num_experts,
            top_k,
            normalize_weights: false,
            scaling_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_softmax_topk_routing() -> Result<()> {
        let device = Device::Cpu;
        // 4 tokens, 8 experts
        let logits = Tensor::randn(0.0f32, 1.0, (4, 8), &device)?;
        let config = RoutingConfig::new_normalized(8, 2);

        let output = SoftmaxTopK::route(&logits, &config)?;

        // Check shapes
        assert_eq!(output.weights.dims(), &[4, 2]);
        assert_eq!(output.indices.dims(), &[4, 2]);

        // Check weights sum to ~1 (normalized)
        let weight_sums = output.weights.sum_keepdim(D::Minus1)?;
        let weight_sums: Vec<f32> = weight_sums.flatten_all()?.to_vec1()?;
        for sum in weight_sums {
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Weights should sum to 1, got {}",
                sum
            );
        }

        Ok(())
    }

    #[test]
    fn test_group_limited_greedy_routing() -> Result<()> {
        let device = Device::Cpu;
        // 4 tokens, 8 experts, 4 groups (2 experts per group)
        let logits = Tensor::randn(0.0f32, 1.0, (4, 8), &device)?;
        let config = RoutingConfig::new_normalized(8, 2);
        let strategy = GroupLimitedGreedy::new(4, 2); // 4 groups, top 2 groups

        let output = strategy.route_with_config(&logits, &config)?;

        // Check shapes
        assert_eq!(output.weights.dims(), &[4, 2]);
        assert_eq!(output.indices.dims(), &[4, 2]);

        // Indices should be valid expert indices (0-7)
        let indices: Vec<u32> = output.indices.flatten_all()?.to_vec1()?;
        for idx in indices {
            assert!(idx < 8, "Index {} out of bounds", idx);
        }

        Ok(())
    }

    #[test]
    fn test_sparsemixer_routing() -> Result<()> {
        let device = Device::Cpu;
        // 4 tokens, 8 experts
        let logits = Tensor::randn(0.0f32, 1.0, (4, 8), &device)?;
        let config = RoutingConfig::new_normalized(8, 2);
        let strategy = SparseMixer::new(0.01); // Small jitter epsilon

        let output = strategy.route_with_config(&logits, &config)?;

        // Check shapes - SparseMixer always returns 2 experts
        assert_eq!(output.weights.dims(), &[4, 2]);
        assert_eq!(output.indices.dims(), &[4, 2]);

        // Indices should be valid expert indices (0-7)
        let indices: Vec<u32> = output.indices.flatten_all()?.to_vec1()?;
        for idx in indices {
            assert!(idx < 8, "Index {} out of bounds", idx);
        }

        // Top-1 and top-2 should be different for each token
        let indices_2d: Vec<Vec<u32>> = output.indices.to_vec2()?;
        for row in indices_2d {
            assert_ne!(
                row[0], row[1],
                "Top-1 and top-2 should be different experts"
            );
        }

        // Weights should be positive
        let weights: Vec<f32> = output.weights.flatten_all()?.to_vec1()?;
        for w in weights {
            assert!(w >= 0.0, "Weight should be non-negative, got {}", w);
        }

        Ok(())
    }
}
