//! Pipeline hooks for stage-level activation streaming.
//!
//! This module provides a hook mechanism for pipeline parallelism,
//! enabling activation streaming between pipeline stages. Hooks
//! intercept activations at stage boundaries (not per-layer) for:
//! - Pipeline parallelism (forwarding activations to remote workers)
//! - Logits return from last stage to first stage
//! - Request lifecycle management

use candle_core::{Result, Tensor};
use std::sync::Arc;

/// Result of receiving from the pipeline activation channel.
///
/// This enum distinguishes between receiving activation data (continue processing)
/// and receiving a completion signal (request finished normally).
#[derive(Debug)]
pub enum ActivationResult {
    /// Got activation data to process - continue with forward pass.
    ///
    /// The tensor shape encodes the phase:
    ///   - seq_len > 1: Prefill phase (accumulate in cache, no logits)
    ///   - seq_len == 1: Decode phase (process and return logits)
    ///
    /// Position is derived from cache length (stream cursor model):
    /// `position = starting_position + cache.len()`
    Data {
        /// Activation tensor [batch, seq_len, hidden_dim].
        tensor: Tensor,
    },
    /// Request completed successfully - no more activations will arrive.
    /// The sequence should transition to Done state, not Error.
    Completed {
        /// Why the request finished (Length, Eos, etc.)
        reason: crate::StopReason,
    },
}

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
    /// * `starting_position` - RoPE starting position (prefix cache offset from HEAD)
    fn init_pipeline_request(
        &self,
        _request_id: uuid::Uuid,
        _total_prompt_tokens: usize,
        _starting_position: usize,
    ) {
        // Default: no-op
    }

    /// Signal that a pipeline request has stopped.
    ///
    /// Called when a sequence transitions to Done state. For pipeline parallelism,
    /// HEAD calls this to notify TAIL that the request is complete.
    ///
    /// This is async because it may send an RPC to downstream stages.
    /// Returns a `'static` future - implementations must clone/own any data they need.
    fn stop_request(
        &self,
        _request_id: uuid::Uuid,
        _reason: crate::sequence::StopReason,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>> {
        Box::pin(std::future::ready(()))
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
    /// # Arguments
    /// * `request_id` - The request ID to receive activations for. Used to route
    ///   to the correct per-request channel in distributed implementations.
    ///
    /// # Returns
    /// - `Ok(ActivationResult::Data { tensor, tokens })` - Got activation, continue processing
    /// - `Ok(ActivationResult::Completed { reason })` - Request finished normally
    /// - `Err(...)` - Actual error (channel closed unexpectedly, etc.)
    ///
    /// Default implementation returns an error. Override in distributed hooks that
    /// need to receive activations from the previous stage.
    fn receive_activation(&self, _request_id: uuid::Uuid) -> Result<ActivationResult> {
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

    /// Call the hook's init_pipeline_request if present.
    ///
    /// This should be called once per request on the first prefill chunk
    /// to initialize pipeline parallelism metadata.
    ///
    /// # Parameters
    /// * `request_id` - Request UUID for correlation
    /// * `total_prompt_tokens` - Total tokens in the complete prompt
    /// * `starting_position` - RoPE starting position (prefix cache offset)
    pub fn call_init_pipeline_request(
        &self,
        request_id: uuid::Uuid,
        total_prompt_tokens: usize,
        starting_position: usize,
    ) {
        if let Some(hook) = &self.hook {
            hook.init_pipeline_request(request_id, total_prompt_tokens, starting_position);
        }
    }

    /// Signal that a pipeline request has stopped.
    ///
    /// Called when a sequence transitions to Done state to notify downstream
    /// pipeline stages. No-op future if no hook is configured.
    pub fn stop_request(
        &self,
        request_id: uuid::Uuid,
        reason: crate::sequence::StopReason,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>> {
        match &self.hook {
            Some(hook) => hook.stop_request(request_id, reason),
            None => Box::pin(std::future::ready(())),
        }
    }

    /// Check if the hook needs external logits for sampling.
    ///
    /// Returns true for PP first stage that receives logits from the last stage.
    pub fn needs_external_logits(&self) -> bool {
        self.hook
            .as_ref()
            .is_some_and(|h| h.needs_external_logits())
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
    /// # Arguments
    /// * `request_id` - The request ID to receive activations for. Used to route
    ///   to the correct per-request channel in distributed implementations.
    ///
    /// # Returns
    /// - `Ok(None)` - No hook configured or this is the first stage
    /// - `Ok(Some(ActivationResult::Data { tensor, tokens }))` - Got activation, continue
    /// - `Ok(Some(ActivationResult::Completed { reason }))` - Request finished normally
    /// - `Err(...)` - Actual error (channel closed unexpectedly, etc.)
    pub fn receive_stage_input(&self, request_id: uuid::Uuid) -> Result<Option<ActivationResult>> {
        match &self.hook {
            Some(hook) => {
                // If layer_range starts at 0, this is first stage - no input to receive
                if hook.layer_range().start == 0 {
                    return Ok(None);
                }
                Ok(Some(hook.receive_activation(request_id)?))
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

    #[test]
    fn test_hook_container_none() {
        let container = HookContainer::none();
        assert!(!container.is_some());
        assert!(container.layer_range().is_none());
    }

    #[test]
    fn test_hook_container_is_first_last_stage() {
        // No hook = always first and last stage
        let container = HookContainer::none();
        assert!(container.is_first_stage());
        assert!(container.is_last_stage());
    }
}
