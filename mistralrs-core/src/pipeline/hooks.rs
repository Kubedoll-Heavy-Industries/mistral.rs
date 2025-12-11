//! Pipeline hooks for layer-by-layer activation interception.
//!
//! This module provides a hook mechanism for intercepting activations during
//! transformer layer processing. This is useful for:
//! - Pipeline parallelism (forwarding activations to remote workers)
//! - Activation analysis and debugging
//! - Layer-wise quantization studies
//!
//! # Example
//!
//! ```ignore
//! use mistralrs_core::pipeline::{PipelineHook, LayerActivation};
//!
//! struct ActivationLogger;
//!
//! impl PipelineHook for ActivationLogger {
//!     fn on_layer_output(
//!         &self,
//!         layer_idx: usize,
//!         activation: &LayerActivation,
//!     ) -> candle_core::Result<Option<candle_core::Tensor>> {
//!         println!("Layer {}: shape {:?}", layer_idx, activation.hidden_states.dims());
//!         Ok(None) // Don't replace activation
//!     }
//! }
//! ```

use candle_core::{Result, Tensor};
use std::sync::Arc;

/// Activation data passed to pipeline hooks.
#[derive(Debug, Clone)]
pub struct LayerActivation<'a> {
    /// Hidden states tensor [batch, seq_len, hidden_dim].
    pub hidden_states: &'a Tensor,
    /// Layer index (0-indexed).
    pub layer_idx: usize,
    /// Total number of layers in the model.
    pub total_layers: usize,
}

/// Hook for intercepting layer activations during forward pass.
///
/// Implement this trait to receive callbacks after each transformer layer
/// processes its input. The hook can optionally replace the activation
/// with a new tensor (useful for pipeline parallelism where a remote
/// worker computes the next stage).
pub trait PipelineHook: Send + Sync {
    /// Called after a transformer layer produces its output.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer that just executed (0-indexed)
    /// * `activation` - The layer's output activation
    ///
    /// # Returns
    /// * `Ok(None)` - Continue with the original activation
    /// * `Ok(Some(tensor))` - Replace activation with the provided tensor
    /// * `Err(e)` - Abort forward pass with error
    ///
    /// # Pipeline Parallelism
    /// For distributed inference, the hook can:
    /// 1. Forward the activation to a remote worker
    /// 2. Wait for the remote worker's response
    /// 3. Return `Some(response_tensor)` to continue with remote result
    ///
    /// To stop processing at a certain layer (hand off to remote),
    /// return an error or use a sentinel value that the caller checks.
    fn on_layer_output(
        &self,
        layer_idx: usize,
        activation: &LayerActivation,
    ) -> Result<Option<Tensor>>;

    /// Called before a transformer layer processes its input.
    ///
    /// Default implementation does nothing. Override if you need
    /// to intercept or modify inputs before layer processing.
    fn on_layer_input(
        &self,
        _layer_idx: usize,
        _activation: &LayerActivation,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Returns the layer range this hook is interested in.
    ///
    /// Default is all layers (0..usize::MAX). Override to filter
    /// which layers trigger the hook for performance optimization.
    fn layer_range(&self) -> std::ops::Range<usize> {
        0..usize::MAX
    }

    /// Whether the hook should be called during prefill phase.
    ///
    /// Default is true. Set to false if you only want to intercept
    /// during token generation (decode) phase.
    fn during_prefill(&self) -> bool {
        true
    }

    /// Whether the hook should be called during decode phase.
    ///
    /// Default is true. Set to false if you only want to intercept
    /// during prefill (prompt processing) phase.
    fn during_decode(&self) -> bool {
        true
    }
}

/// A no-op hook that does nothing (for when hooks are disabled).
pub struct NoOpHook;

impl PipelineHook for NoOpHook {
    fn on_layer_output(
        &self,
        _layer_idx: usize,
        _activation: &LayerActivation,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

/// Wrapper for optional pipeline hooks.
///
/// This provides a convenient way to conditionally call hooks
/// without Option checks everywhere.
#[derive(Clone)]
pub struct HookContainer {
    hook: Option<Arc<dyn PipelineHook>>,
}

impl HookContainer {
    /// Create a new container with the given hook.
    pub fn new(hook: Arc<dyn PipelineHook>) -> Self {
        Self { hook: Some(hook) }
    }

    /// Create an empty container (no hook).
    pub fn none() -> Self {
        Self { hook: None }
    }

    /// Check if a hook is present.
    pub fn is_some(&self) -> bool {
        self.hook.is_some()
    }

    /// Call the hook's on_layer_output if present.
    ///
    /// Returns the original activation if no hook or hook returns None.
    pub fn call_layer_output(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        total_layers: usize,
    ) -> Result<Option<Tensor>> {
        match &self.hook {
            Some(hook) if hook.layer_range().contains(&layer_idx) => {
                let activation = LayerActivation {
                    hidden_states,
                    layer_idx,
                    total_layers,
                };
                hook.on_layer_output(layer_idx, &activation)
            }
            _ => Ok(None),
        }
    }

    /// Call the hook's on_layer_input if present.
    pub fn call_layer_input(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        total_layers: usize,
    ) -> Result<Option<Tensor>> {
        match &self.hook {
            Some(hook) if hook.layer_range().contains(&layer_idx) => {
                let activation = LayerActivation {
                    hidden_states,
                    layer_idx,
                    total_layers,
                };
                hook.on_layer_input(layer_idx, &activation)
            }
            _ => Ok(None),
        }
    }
}

impl Default for HookContainer {
    fn default() -> Self {
        Self::none()
    }
}

impl std::fmt::Debug for HookContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.hook {
            Some(_) => write!(f, "HookContainer(Some)"),
            None => write!(f, "HookContainer(None)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    struct CountingHook {
        count: std::sync::atomic::AtomicUsize,
    }

    impl PipelineHook for CountingHook {
        fn on_layer_output(
            &self,
            _layer_idx: usize,
            _activation: &LayerActivation,
        ) -> Result<Option<Tensor>> {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(None)
        }
    }

    #[test]
    fn test_hook_container_none() {
        let container = HookContainer::none();
        assert!(!container.is_some());

        let tensor = Tensor::zeros((1, 4, 8), candle_core::DType::F32, &Device::Cpu).unwrap();
        let result = container.call_layer_output(0, &tensor, 32).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_hook_container_with_hook() {
        let hook = Arc::new(CountingHook {
            count: std::sync::atomic::AtomicUsize::new(0),
        });
        let container = HookContainer::new(hook.clone());
        assert!(container.is_some());

        let tensor = Tensor::zeros((1, 4, 8), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Call hook 3 times
        container.call_layer_output(0, &tensor, 32).unwrap();
        container.call_layer_output(1, &tensor, 32).unwrap();
        container.call_layer_output(2, &tensor, 32).unwrap();

        assert_eq!(hook.count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn test_layer_range_filtering() {
        struct RangedHook {
            count: std::sync::atomic::AtomicUsize,
        }

        impl PipelineHook for RangedHook {
            fn on_layer_output(
                &self,
                _layer_idx: usize,
                _activation: &LayerActivation,
            ) -> Result<Option<Tensor>> {
                self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(None)
            }

            fn layer_range(&self) -> std::ops::Range<usize> {
                5..10 // Only layers 5-9
            }
        }

        let hook = Arc::new(RangedHook {
            count: std::sync::atomic::AtomicUsize::new(0),
        });
        let container = HookContainer::new(hook.clone());

        let tensor = Tensor::zeros((1, 4, 8), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Call for layers 0-14
        for i in 0..15 {
            container.call_layer_output(i, &tensor, 32).unwrap();
        }

        // Only layers 5-9 should have triggered (5 calls)
        assert_eq!(hook.count.load(std::sync::atomic::Ordering::SeqCst), 5);
    }
}
