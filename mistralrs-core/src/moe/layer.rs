//! Type-safe MoE layer implementing the FeedForward trait.
//!
//! This module provides `MoE<R>`, a mixture-of-experts layer that:
//! - Uses the `RoutingStrategy` trait for type-safe routing dispatch
//! - Implements `FeedForward` for seamless integration with transformer blocks
//! - Supports optional shared experts (DeepSeek style)
//!
//! # Design
//!
//! The MoE layer encapsulates:
//! - A gate (router projection) that produces logits over experts
//! - `MoEExperts` for the actual expert computation
//! - A `RoutingStrategy` that determines how tokens are routed
//!
//! By parameterizing over the routing strategy, we get:
//! - Compile-time monomorphization (zero runtime overhead for routing dispatch)
//! - Type-level documentation of which strategy a model uses
//! - Impossible to accidentally use the wrong routing strategy

use std::marker::PhantomData;
use std::sync::Arc;

use candle_core::{Device, IndexOp, Result, Tensor};
use mistralrs_quant::{NonZeroOp, QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};

use super::experts::{MoEExperts, MoEExpertsConfig};
use super::routing::{RoutingConfig, RoutingStrategy, SoftmaxTopK, SparseMixer};
use crate::layers::{Activation, FeedForward, Mlp};

// ============================================================================
// MoE Layer Configuration
// ============================================================================

/// Configuration for creating an MoE layer.
#[derive(Debug, Clone)]
pub struct MoELayerConfig {
    /// Hidden size of the model
    pub hidden_size: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts each token is routed to
    pub num_experts_per_tok: usize,
    /// Intermediate size for expert MLPs
    pub moe_intermediate_size: usize,
    /// Whether to normalize top-k routing probabilities
    pub norm_topk_prob: bool,
    /// Scaling factor for routing weights (used when not normalizing)
    pub routed_scaling_factor: f64,
    /// Activation function for expert MLPs
    pub hidden_act: Activation,
    /// Optional quantization config
    pub quantization_config: Option<QuantizedConfig>,
}

impl MoELayerConfig {
    /// Create an MoEExpertsConfig from this layer config.
    pub fn to_experts_config(&self) -> MoEExpertsConfig {
        MoEExpertsConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            hidden_size: self.hidden_size,
            moe_intermediate_size: self.moe_intermediate_size,
        }
    }

    /// Create a RoutingConfig from this layer config.
    pub fn to_routing_config(&self) -> RoutingConfig {
        if self.norm_topk_prob {
            RoutingConfig::new_normalized(self.num_experts, self.num_experts_per_tok)
        } else {
            RoutingConfig::new_scaled(self.num_experts, self.num_experts_per_tok, self.routed_scaling_factor)
        }
    }
}

// ============================================================================
// MoE Layer (Type-Safe over Routing Strategy)
// ============================================================================

/// Mixture of Experts layer with type-safe routing.
///
/// This struct implements `FeedForward`, allowing it to be used as a drop-in
/// replacement for standard MLP layers in transformer blocks.
///
/// # Type Parameters
///
/// - `R`: The routing strategy type (e.g., `SoftmaxTopK`)
///
/// # Example
///
/// ```ignore
/// // Create a Qwen3-style MoE layer with softmax top-k routing
/// let moe: MoE<SoftmaxTopK> = MoE::new(config, vb, device, comm, loading_isq)?;
///
/// // Use in a transformer block
/// let output = moe.forward(&hidden_states)?;
/// ```
pub struct MoE<R: RoutingStrategy = SoftmaxTopK> {
    /// Gate (router) projection: hidden_size -> num_experts
    /// Uses Arc<dyn QuantMethod> for consistent ISQ support across all weight formats
    gate: Arc<dyn QuantMethod>,
    /// Expert networks
    experts: MoEExperts,
    /// Routing configuration
    routing_config: RoutingConfig,
    /// Optional shared expert (DeepSeek style)
    shared_expert: Option<Mlp>,
    /// Phantom data for the routing strategy type
    _routing: PhantomData<R>,
}

impl<R: RoutingStrategy> MoE<R> {
    /// Create a new MoE layer from safetensors weights.
    ///
    /// # Arguments
    ///
    /// * `config` - MoE layer configuration
    /// * `vb` - Variable builder for loading weights
    /// * `layer_device` - Device for this layer
    /// * `comm` - Communication handle for tensor parallelism
    /// * `loading_isq` - Whether ISQ (in-situ quantization) is being used
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &MoELayerConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        // Load gate (router) using ReplicatedLayer for consistent ISQ support
        let gate = ReplicatedLayer::new(
            config.hidden_size,
            config.num_experts,
            &config.quantization_config,
            false, // no bias
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        // Load experts
        let experts_config = config.to_experts_config();
        let experts = MoEExperts::new(
            &experts_config,
            vb,
            layer_device,
            comm,
            loading_isq,
            &config.quantization_config,
            config.hidden_act,
        )?;

        let routing_config = config.to_routing_config();

        Ok(Self {
            gate,
            experts,
            routing_config,
            shared_expert: None,
            _routing: PhantomData,
        })
    }

    /// Create a new MoE layer from pre-constructed components.
    ///
    /// This is used for GGUF loading where the gate is already an `Arc<dyn QuantMethod>`.
    pub fn from_parts(
        gate: Arc<dyn QuantMethod>,
        experts: MoEExperts,
        routing_config: RoutingConfig,
    ) -> Self {
        Self {
            gate,
            experts,
            routing_config,
            shared_expert: None,
            _routing: PhantomData,
        }
    }

    /// Create a new MoE layer with a shared expert.
    ///
    /// Shared experts receive all tokens and their output is added to the
    /// routed expert output. This is used by models like DeepSeek V2.
    #[allow(clippy::too_many_arguments)]
    pub fn with_shared_expert(mut self, shared_expert: Mlp) -> Self {
        self.shared_expert = Some(shared_expert);
        self
    }

    /// Get a reference to the gate for residual tensor serialization.
    pub fn gate(&self) -> &Arc<dyn QuantMethod> {
        &self.gate
    }

    /// Get mutable references to ISQ layers for quantization.
    /// Includes both the gate and expert layers.
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = vec![&mut self.gate];
        layers.extend(self.experts.get_isq_layers());
        layers
    }
}

impl<R: RoutingStrategy> FeedForward for MoE<R> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        // 1. Compute router logits
        let router_logits = self.gate.forward(&xs_flat)?;

        // 2. Route using the type-parameterized strategy
        let route_output = R::route(&router_logits, &self.routing_config)?;

        // 3. Forward through experts
        let mut output = self.experts.forward(xs, route_output.weights, &route_output.indices)?;

        // 4. Add shared expert output if present
        if let Some(ref shared) = self.shared_expert {
            let shared_output = shared.forward(xs)?;
            output = (output + shared_output)?;
        }

        output.reshape((b_size, seq_len, hidden_dim))
    }
}

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

/// MoE with standard softmax top-k routing (Qwen3, Mixtral style).
pub type SoftmaxMoE = MoE<SoftmaxTopK>;

// Note: GroupLimitedGreedy MoE requires instance-level configuration for
// n_groups and topk_groups, so it uses a separate type below.

// ============================================================================
// MoE with Group-Limited Routing (DeepSeek style)
// ============================================================================

use super::routing::GroupLimitedGreedy;

/// MoE layer with group-limited greedy routing.
///
/// This is used by DeepSeek V2 style models where experts are divided
/// into groups and routing is limited to selecting top-k groups first.
pub struct GroupLimitedMoE {
    /// Gate (router) projection
    gate: Arc<dyn QuantMethod>,
    /// Expert networks
    experts: MoEExperts,
    /// Routing configuration
    routing_config: RoutingConfig,
    /// Group routing strategy
    group_strategy: GroupLimitedGreedy,
    /// Optional shared expert
    shared_expert: Option<Mlp>,
}

impl GroupLimitedMoE {
    /// Create a new GroupLimitedMoE layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &MoELayerConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        n_groups: usize,
        topk_groups: usize,
    ) -> Result<Self> {
        // Load gate using ReplicatedLayer for consistent ISQ support
        let gate = ReplicatedLayer::new(
            config.hidden_size,
            config.num_experts,
            &config.quantization_config,
            false, // no bias
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        // Load experts
        let experts_config = config.to_experts_config();
        let experts = MoEExperts::new(
            &experts_config,
            vb,
            layer_device,
            comm,
            loading_isq,
            &config.quantization_config,
            config.hidden_act,
        )?;

        let routing_config = config.to_routing_config();
        let group_strategy = GroupLimitedGreedy::new(n_groups, topk_groups);

        Ok(Self {
            gate,
            experts,
            routing_config,
            group_strategy,
            shared_expert: None,
        })
    }

    /// Add a shared expert.
    pub fn with_shared_expert(mut self, shared_expert: Mlp) -> Self {
        self.shared_expert = Some(shared_expert);
        self
    }

    /// Get a reference to the gate.
    pub fn gate(&self) -> &Arc<dyn QuantMethod> {
        &self.gate
    }

    /// Get mutable references to ISQ layers.
    /// Includes both the gate and expert layers.
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = vec![&mut self.gate];
        layers.extend(self.experts.get_isq_layers());
        layers
    }
}

impl FeedForward for GroupLimitedMoE {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        // 1. Compute router logits
        let router_logits = self.gate.forward(&xs_flat)?;

        // 2. Route using group-limited strategy
        let route_output = self.group_strategy.route_with_config(&router_logits, &self.routing_config)?;

        // 3. Forward through experts
        let mut output = self.experts.forward(xs, route_output.weights, &route_output.indices)?;

        // 4. Add shared expert output if present
        if let Some(ref shared) = self.shared_expert {
            let shared_output = shared.forward(xs)?;
            output = (output + shared_output)?;
        }

        output.reshape((b_size, seq_len, hidden_dim))
    }
}

// ============================================================================
// SparseMixer MoE (Phi-3.5 style)
// ============================================================================

/// MoE layer with SparseMixer routing (Phi-3.5 style).
///
/// This MoE variant uses jittered threshold-based masking for top-2 expert
/// selection. It always selects exactly 2 experts per token.
///
/// # Differences from Standard MoE
///
/// - Uses `SparseMixer` routing (argmax + threshold masking)
/// - Always selects 2 experts (top_k is fixed)
/// - Uses jitter epsilon for threshold computation
/// - Individual expert MLPs (not unified MoEExperts backend)
///
/// # Note
///
/// This implementation uses individual `Mlp` experts rather than the unified
/// `MoEExperts` backend because Phi-3.5 MoE uses a CPU-based routing loop
/// that is incompatible with the fused expert backends.
pub struct SparseMixerMoE {
    /// Gate (router) projection
    gate: Arc<dyn QuantMethod>,
    /// Individual expert MLPs
    experts: Vec<Mlp>,
    /// SparseMixer routing strategy
    sparse_mixer: SparseMixer,
    /// Routing configuration
    routing_config: RoutingConfig,
    /// Number of experts
    num_experts: usize,
}

impl SparseMixerMoE {
    /// Create a new SparseMixerMoE layer.
    ///
    /// # Arguments
    ///
    /// * `config` - MoE layer configuration
    /// * `vb` - Variable builder for loading weights
    /// * `layer_device` - Device for this layer
    /// * `comm` - Communication handle for tensor parallelism
    /// * `jitter_eps` - Jitter epsilon for SparseMixer routing
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &MoELayerConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        jitter_eps: f64,
    ) -> Result<Self> {
        // Load gate (router) using ReplicatedLayer for consistent ISQ support
        let gate = ReplicatedLayer::new(
            config.hidden_size,
            config.num_experts,
            &config.quantization_config,
            false, // no bias
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        // Load individual expert MLPs
        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            experts.push(Mlp::new(
                experts_vb.pp(i).set_device(layer_device.clone()),
                config.hidden_size,
                config.moe_intermediate_size,
                &config.quantization_config,
                config.hidden_act,
                comm,
            )?);
        }

        let routing_config = config.to_routing_config();
        let sparse_mixer = SparseMixer::new(jitter_eps);

        Ok(Self {
            gate,
            experts,
            sparse_mixer,
            routing_config,
            num_experts: config.num_experts,
        })
    }

    /// Get a reference to the gate.
    pub fn gate(&self) -> &Arc<dyn QuantMethod> {
        &self.gate
    }

    /// Get mutable references to ISQ layers.
    /// Includes both the gate and expert layers.
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = vec![&mut self.gate];
        for expert in &mut self.experts {
            layers.push(&mut expert.gate);
            layers.push(&mut expert.up);
            layers.push(&mut expert.down);
        }
        layers
    }
}

impl FeedForward for SparseMixerMoE {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        // 1. Compute router logits
        let router_logits = self.gate.forward(&xs_flat)?;

        // 2. Route using SparseMixer strategy
        let route_output = self.sparse_mixer.route_with_config(&router_logits, &self.routing_config)?;

        // 3. Create one-hot expert mask for efficient indexing
        let experts_mask = candle_nn::encoding::one_hot(
            route_output.indices.clone(),
            self.num_experts,
            1u8,
            0u8,
        )?
        .permute((2, 1, 0))?; // [num_experts, top_k, batch*seq]

        // 4. Process each expert
        let mut final_hidden_states = xs_flat.zeros_like()?;

        for expert_idx in 0..self.num_experts {
            let expert = &self.experts[expert_idx];
            let expert_mask = experts_mask.i(expert_idx)?;

            // Find tokens routed to this expert
            let nonzero_mask = expert_mask.contiguous()?.nonzero()?;
            if nonzero_mask.dim(0)? == 0 {
                continue;
            }

            let idx = nonzero_mask.i((.., 0))?; // Which top-k slot (0 or 1)
            let top_x = nonzero_mask.i((.., 1))?; // Which token

            // Get hidden states for tokens routed to this expert
            let current_state = xs_flat.index_select(&top_x, 0)?;

            // Get routing weights for these tokens
            let current_routing_weights = route_output
                .weights
                .index_select(&top_x, 0)?
                .gather(&idx.unsqueeze(1)?.contiguous()?, 1)?;

            // Forward through expert
            let exp_out = expert.forward(&current_state.unsqueeze(0)?)?;
            let current_hidden_states = exp_out.squeeze(0)?.broadcast_mul(&current_routing_weights)?;

            // Accumulate results
            final_hidden_states = final_hidden_states.index_add(
                &top_x.contiguous()?,
                &current_hidden_states.to_dtype(xs_flat.dtype())?,
                0,
            )?;
        }

        final_hidden_states.reshape((b_size, seq_len, hidden_dim))
    }
}

// ============================================================================
// MoE or MLP Enum (For Models with Mixed Layers)
// ============================================================================

/// Enum for layers that can be either MoE or standard MLP.
///
/// Many MoE models (Qwen3 MoE, DeepSeek V2) have some layers that are
/// standard dense layers while others are MoE. This enum provides a
/// unified interface.
///
/// # Type Parameter
///
/// - `R`: Routing strategy for the MoE variant
pub enum MoEOrMlp<R: RoutingStrategy = SoftmaxTopK> {
    /// MoE layer
    MoE(MoE<R>),
    /// Standard MLP layer
    Mlp(Mlp),
}

impl<R: RoutingStrategy> FeedForward for MoEOrMlp<R> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE(moe) => moe.forward(xs),
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

impl<R: RoutingStrategy> MoEOrMlp<R> {
    /// Get ISQ layers from this layer (MoE or MLP).
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            Self::MoE(moe) => moe.get_isq_layers(),
            Self::Mlp(mlp) => {
                // Access Mlp fields directly since they're public
                vec![&mut mlp.gate, &mut mlp.up, &mut mlp.down]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_moe_layer_config() {
        let config = MoELayerConfig {
            hidden_size: 4096,
            num_experts: 8,
            num_experts_per_tok: 2,
            moe_intermediate_size: 11008,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            hidden_act: Activation::Silu,
            quantization_config: None,
        };

        let experts_config = config.to_experts_config();
        assert_eq!(experts_config.num_experts, 8);
        assert_eq!(experts_config.num_experts_per_tok, 2);

        let routing_config = config.to_routing_config();
        assert_eq!(routing_config.num_experts, 8);
        assert_eq!(routing_config.top_k, 2);
        assert!(routing_config.normalize_weights);
    }
}
