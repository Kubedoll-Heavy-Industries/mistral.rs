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
    /// Token sequence for this forward pass.
    ///
    /// For pipeline parallelism: hooks can extract tokens and propagate them
    /// with activations to enable sparse KV cache reconstruction on remote workers.
    ///
    /// During prefill: contains tokens for the current chunk being processed
    /// During decode: contains the current token only (due to model forward() API)
    pub tokens: &'a [u32],
    /// Request ID (UUID7) for correlation across pipeline stages.
    ///
    /// Generated once per request via `MistralRs::next_request_id()`.
    /// Enables distributed tracing and activation correlation in pipeline parallelism.
    pub request_id: uuid::Uuid,
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

    /// Whether this hook provides external logits for sampling.
    ///
    /// Returns true for PP first stage (but not last stage) that receives
    /// logits from the last stage via transport layer instead of computing
    /// them locally.
    ///
    /// When this returns true, the pipeline should call `receive_response_logits()`
    /// after forward pass to get logits for sampling instead of using the
    /// local forward() result.
    fn needs_external_logits(&self) -> bool {
        false
    }

    /// Receive response logits from the last pipeline stage.
    ///
    /// This is called by the pipeline code on the first stage (but not last)
    /// to get logits for sampling instead of using the local forward() result.
    ///
    /// Blocks until logits arrive from the transport layer.
    ///
    /// Default implementation returns an error. Override in hooks that
    /// support external logits (i.e., where `needs_external_logits()` returns true).
    ///
    /// # Returns
    /// - `Ok(Tensor)` - The logits tensor for sampling [batch, seq_len, vocab_size]
    /// - `Err` - If not configured for logits or channel closed
    fn receive_response_logits(&self) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "receive_response_logits not supported by this hook".to_string(),
        ))
    }

    /// Set pending tokens for sparse KV cache propagation (pipeline parallelism).
    ///
    /// Called before the forward pass begins when this is a pipeline continuation
    /// request. Allows hooks to store the token sequence for later extraction
    /// and propagation with activations.
    ///
    /// Default implementation does nothing. Override in distributed hooks that
    /// need to propagate tokens for sparse KV cache reconstruction.
    fn set_pending_tokens(&self, _tokens: Vec<u32>) {
        // Default: no-op
    }

    /// Set request context for the current forward pass.
    ///
    /// This MUST be called before forward() begins to establish the request context
    /// for activation sending/receiving. Hooks use this stored context instead of
    /// threading request_id through model forward signatures.
    ///
    /// # Parameters
    /// * `request_id` - The request UUID (UUID7) for correlation
    fn set_request_context(&self, _request_id: uuid::Uuid) {
        // Default: no-op
    }

    /// Send init RPC for pipeline parallelism (called once per request).
    ///
    /// This method MUST be called exactly once per request BEFORE the first
    /// `send_activation()` call. It informs all downstream pipeline stages about
    /// the total prompt length, enabling them to correctly detect prefill vs decode
    /// boundaries during chunked processing.
    ///
    /// # Separation of Concerns
    /// - `init_pipeline_request()`: One-time metadata setup (total_prompt_tokens)
    /// - `send_activation()`: Per-chunk/per-token data streaming
    ///
    /// Default implementation does nothing. Override in distributed hooks that
    /// support pipeline parallelism.
    ///
    /// # Parameters
    /// * `request_id` - The request UUID (UUID7) for correlation
    /// * `total_prompt_tokens` - Total tokens in the complete prompt (not per-chunk)
    fn init_pipeline_request(&self, _request_id: uuid::Uuid, _total_prompt_tokens: usize) {
        // Default: no-op
    }

    /// Send activation to next pipeline stage (called by pipeline after forward).
    ///
    /// This is called by the pipeline orchestration layer after forward() completes
    /// on stages that are not the last stage. Should be no-op on last stage.
    ///
    /// # Important
    /// Before calling this for the first time, you MUST call `init_pipeline_request()`
    /// to send initialization metadata. This method only sends per-chunk/per-token data.
    ///
    /// Default implementation does nothing. Override in distributed hooks that
    /// need to send activations to the next stage.
    ///
    /// # Parameters
    /// * `hidden` - The activation tensor to send
    /// * `tokens` - Token sequence for this forward pass
    /// * `request_id` - Request UUID for correlation
    /// * `sequence_position` - RoPE position offset for this chunk (from seqlen_offsets)
    fn send_activation(
        &self,
        _hidden: &Tensor,
        _tokens: &[u32],
        _request_id: uuid::Uuid,
        _sequence_position: usize,
    ) -> Result<()> {
        // Default: no-op
        Ok(())
    }

    /// Receive activation from previous pipeline stage (called by pipeline before forward).
    ///
    /// This is called by the pipeline orchestration layer before forward() begins
    /// on stages that are not the first stage. Should error on first stage.
    ///
    /// Default implementation returns an error. Override in distributed hooks that
    /// need to receive activations from the previous stage.
    fn receive_activation(&self) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "receive_activation not supported by this hook".to_string(),
        ))
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

    /// Get a reference to the contained hook, if any.
    pub fn get(&self) -> Option<&Arc<dyn PipelineHook>> {
        self.hook.as_ref()
    }

    /// Call the hook's on_layer_output if present.
    ///
    /// Returns the original activation if no hook or hook returns None.
    pub fn call_layer_output(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        total_layers: usize,
        tokens: &[u32],
        request_id: uuid::Uuid,
    ) -> Result<Option<Tensor>> {
        match &self.hook {
            Some(hook) if hook.layer_range().contains(&layer_idx) => {
                let activation = LayerActivation {
                    hidden_states,
                    layer_idx,
                    total_layers,
                    tokens,
                    request_id,
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
        tokens: &[u32],
        request_id: uuid::Uuid,
    ) -> Result<Option<Tensor>> {
        match &self.hook {
            Some(hook) if hook.layer_range().contains(&layer_idx) => {
                let activation = LayerActivation {
                    hidden_states,
                    layer_idx,
                    total_layers,
                    tokens,
                    request_id,
                };
                hook.on_layer_input(layer_idx, &activation)
            }
            _ => Ok(None),
        }
    }

    /// Call the hook's init_pipeline_request if present.
    ///
    /// This should be called once per request on the first prefill chunk
    /// to initialize pipeline parallelism metadata.
    pub fn call_init_pipeline_request(&self, request_id: uuid::Uuid, total_prompt_tokens: usize) {
        if let Some(hook) = &self.hook {
            hook.init_pipeline_request(request_id, total_prompt_tokens);
        }
    }

    /// Check if the hook needs external logits for sampling.
    ///
    /// Returns true for PP first stage that receives logits from the last stage.
    pub fn needs_external_logits(&self) -> bool {
        self.hook.as_ref().is_some_and(|h| h.needs_external_logits())
    }

    /// Receive response logits from the last pipeline stage.
    ///
    /// Blocks until logits arrive. Only valid when `needs_external_logits()` is true.
    pub fn receive_response_logits(&self) -> Result<Tensor> {
        match &self.hook {
            Some(hook) if hook.needs_external_logits() => hook.receive_response_logits(),
            Some(_) => Err(candle_core::Error::Msg(
                "Hook does not support external logits".to_string(),
            )),
            None => Err(candle_core::Error::Msg("No hook configured".to_string())),
        }
    }

    /// Get the layer range this hook handles.
    ///
    /// Used by pipeline to determine which layer indices to pass to hook calls.
    /// Returns None if no hook is configured.
    pub fn layer_range(&self) -> Option<std::ops::Range<usize>> {
        self.hook.as_ref().map(|h| h.layer_range())
    }

    // === Stage-level operations (preferred API for pipeline parallelism) ===

    /// Set request context for the current forward pass.
    ///
    /// MUST be called before forward() to establish context for activation
    /// sending/receiving. Hooks use stored context instead of threading
    /// request_id through model forward signatures.
    pub fn set_request_context(&self, request_id: uuid::Uuid) {
        if let Some(hook) = &self.hook {
            hook.set_request_context(request_id);
        }
    }

    /// Receive activation from previous pipeline stage (STAGE-LEVEL).
    ///
    /// Called BEFORE forward_pass() on non-first stages.
    /// Blocks until activation arrives from the transport layer.
    ///
    /// Returns None if no hook configured or this is the first stage.
    pub fn receive_stage_input(&self) -> Result<Option<Tensor>> {
        match &self.hook {
            Some(hook) => {
                // If layer_range starts at 0, this is first stage - no input to receive
                if hook.layer_range().start == 0 {
                    return Ok(None);
                }
                Ok(Some(hook.receive_activation()?))
            }
            None => Ok(None),
        }
    }

    /// Send activation to next pipeline stage (STAGE-LEVEL).
    ///
    /// Called AFTER forward_pass() on non-last stages.
    /// Does nothing if no hook configured or this is the last stage.
    ///
    /// # Parameters
    /// * `hidden` - The activation tensor to send
    /// * `tokens` - Token sequence for this forward pass
    /// * `request_id` - Request UUID for correlation
    /// * `sequence_position` - RoPE position offset for this chunk (from seqlen_offsets[0])
    pub fn send_stage_output(
        &self,
        hidden: &Tensor,
        tokens: &[u32],
        request_id: uuid::Uuid,
        sequence_position: usize,
    ) -> Result<()> {
        if let Some(hook) = &self.hook {
            hook.send_activation(hidden, tokens, request_id, sequence_position)?;
        }
        Ok(())
    }

    /// Check if this is the first pipeline stage (layer_range starts at 0).
    pub fn is_first_stage(&self) -> bool {
        self.hook
            .as_ref()
            .map(|h| h.layer_range().start == 0)
            .unwrap_or(true)
    }

    /// Check if this is the last pipeline stage.
    ///
    /// Determined by whether the hook needs external logits - first stage (but not last)
    /// needs external logits from the last stage.
    pub fn is_last_stage(&self) -> bool {
        !self.needs_external_logits()
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
        let tokens = vec![];
        let request_id = uuid::Uuid::now_v7();
        let result = container.call_layer_output(0, &tensor, 32, &tokens, request_id).unwrap();
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
        let tokens = vec![];
        let request_id = uuid::Uuid::now_v7();

        // Call hook 3 times
        container.call_layer_output(0, &tensor, 32, &tokens, request_id).unwrap();
        container.call_layer_output(1, &tensor, 32, &tokens, request_id).unwrap();
        container.call_layer_output(2, &tensor, 32, &tokens, request_id).unwrap();

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
        let tokens = vec![];
        let request_id = uuid::Uuid::now_v7();

        // Call for layers 0-14
        for i in 0..15 {
            container.call_layer_output(i, &tensor, 32, &tokens, request_id).unwrap();
        }

        // Only layers 5-9 should have triggered (5 calls)
        assert_eq!(hook.count.load(std::sync::atomic::Ordering::SeqCst), 5);
    }
}
