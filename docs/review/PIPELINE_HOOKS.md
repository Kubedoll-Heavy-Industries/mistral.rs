# Pipeline Hooks Review

**Files**: `mistralrs-core/src/pipeline/hooks.rs`
**Lines Changed**: ~541

## Summary

The pipeline hooks system has been significantly extended to support distributed inference:

- Extended `LayerActivation` struct with `tokens` and `request_id` fields
- New methods on `PipelineHook` trait for PP coordination
- New `AsyncPipelineHook` trait for non-blocking activation processing
- `LayerExecutor` and `Layered` traits for centralized hook orchestration
- Helper functions `execute_with_hooks`, `make_hook_closures`, `iterate_layers`
- Extended `HookContainer` with stage-level operations

## Strengths

### 1. Clean Trait Object Safety
`PipelineHook` trait remains object-safe (no generic methods, no `Self` in argument position). `Arc<dyn PipelineHook>` works correctly.

### 2. Sensible Default Implementations
All new methods have no-op defaults, maintaining backward compatibility.

### 3. Good Separation Between Sync and Async
`AsyncPipelineHook` as separate trait rather than async methods on `PipelineHook`.

### 4. Centralized Hook Orchestration
`execute_with_hooks` (lines 603-639) provides single point for hook invocation.

### 5. Rich Documentation
Extensive doc comments explain purpose, usage, and invariants.

## Issues

### Issue 1: CRITICAL - Duplicate Abstractions
**Location**: Lines 529-564 (`LayerExecutor`) and 704-737 (`Layered`)
**Severity**: High

Two traits serve nearly identical purposes:

```rust
// LayerExecutor
pub trait LayerExecutor {
    type Context;
    fn forward_layer(&self, layer_idx: usize, activation: Tensor, context: &mut Self::Context) -> Result<Tensor>;
    fn num_layers(&self) -> usize;
    fn layer_start(&self) -> usize;
}

// Layered (uses GATs)
pub trait Layered {
    type State<'a> where Self: 'a;
    fn forward_layer<'a>(&'a self, layer_idx: usize, activation: Tensor, state: &mut Self::State<'a>) -> Result<Tensor>;
    fn num_layers(&self) -> usize;
}
```

**Neither is implemented anywhere** (only commented-out code in `quantized_llama.rs`).

**Recommendation**: Pick one and remove the other.

### Issue 2: HIGH - `AsyncPipelineHook` Orphaned
**Location**: Lines 243-264
**Severity**: High

```rust
#[async_trait]
pub trait AsyncPipelineHook: Send + Sync {
    fn try_prefetch_input(&self, layer_idx: usize) -> Result<Option<Tensor>>;
    fn layer_range(&self) -> std::ops::Range<usize> { 0..usize::MAX }
    fn needs_external_logits(&self) -> bool { false }
    async fn receive_response_logits_async(&self) -> Result<Tensor>;
}
```

**Problems:**
- No implementations exist
- No integration with `HookContainer`
- Missing async versions of critical methods

**Recommendation**: Remove or complete with proper integration.

### Issue 3: HIGH - Method Proliferation
**Location**: Lines 62-241
**Severity**: High

`PipelineHook` now has **13 methods**:
- `on_layer_output` (required)
- `on_layer_input`
- `layer_range`
- `during_prefill` / `during_decode`
- `needs_external_logits`
- `receive_response_logits`
- `set_pending_tokens`
- `set_request_context`
- `init_pipeline_request`
- `send_activation`
- `receive_activation`

**Violates Interface Segregation Principle.** Mixes:
- Layer-level interception (`on_layer_*`)
- Stage-level transport (`send_activation`, `receive_activation`)
- Request lifecycle (`set_request_context`, `init_pipeline_request`)
- Phase filtering (`during_prefill`, `during_decode`)

**Recommendation**: Split into focused traits:

```rust
pub trait LayerHook: Send + Sync {
    fn on_layer_output(&self, layer_idx: usize, activation: &LayerActivation) -> Result<Option<Tensor>>;
    fn on_layer_input(&self, layer_idx: usize, activation: &LayerActivation) -> Result<Option<Tensor>>;
    fn layer_range(&self) -> Range<usize>;
}

pub trait PipelineTransport: LayerHook {
    fn send_activation(&self, hidden: &Tensor, tokens: &[u32], request_id: Uuid, seq_pos: usize) -> Result<()>;
    fn receive_activation(&self) -> Result<Tensor>;
    fn receive_response_logits(&self) -> Result<Tensor>;
}

pub trait PipelineLifecycle: Send + Sync {
    fn set_request_context(&self, request_id: Uuid);
    fn init_pipeline_request(&self, request_id: Uuid, total_prompt_tokens: usize);
}
```

### Issue 4: MEDIUM - Inconsistent Error Handling
**Location**: Lines 150-154, 218, 236-240
**Severity**: Medium

```rust
// receive_response_logits - errors when not implemented
fn receive_response_logits(&self) -> Result<Tensor> {
    Err(candle_core::Error::Msg("not supported..."))
}

// send_activation - silently succeeds
fn send_activation(...) -> Result<()> {
    Ok(())  // Silent no-op
}

// receive_activation - errors when not implemented
fn receive_activation(&self) -> Result<Tensor> {
    Err(candle_core::Error::Msg("not supported..."))
}
```

**Recommendation**: Make consistent - all optional methods either error or no-op.

### Issue 5: MEDIUM - Stage Detection Logic Fragile
**Location**: Lines 453-466
**Severity**: Medium

```rust
pub fn is_first_stage(&self) -> bool {
    self.hook.as_ref().map(|h| h.layer_range().start == 0).unwrap_or(true)
}

pub fn is_last_stage(&self) -> bool {
    !self.needs_external_logits()  // Inverted logic
}
```

Different heuristics for same concept. Could fail for dynamic layer loading.

**Recommendation**: Add explicit stage position to hook configuration:

```rust
pub enum StagePosition {
    First, Middle, Last, OnlyStage,
}
```

### Issue 6: MEDIUM - GAT May Limit Adoption
**Location**: Lines 711-714
**Severity**: Medium

```rust
pub trait Layered {
    type State<'a> where Self: 'a;
```

GATs are uncommon and the `Self: 'a` bound confuses many developers. For a trait that isn't implemented anywhere, this may be over-engineered.

### Issue 7: LOW - Unused `during_prefill`/`during_decode`
**Location**: Lines 108-122
**Severity**: Low

Methods exist but are never checked by `HookContainer`.

**Recommendation**: Either wire up phase filtering or remove.

### Issue 8: LOW - Redundant Closures
**Location**: Lines 671-697
**Severity**: Low

`make_hook_closures` creates two nearly identical closures.

## Recommendations Summary

1. **Split `PipelineHook`** into focused traits (LayerHook, PipelineTransport, PipelineLifecycle)
2. **Remove or complete `AsyncPipelineHook`**
3. **Consolidate `LayerExecutor` and `Layered`** - pick one
4. **Make error handling consistent** across methods
5. **Add explicit stage position** instead of inferring from layer_range
6. **Implement or remove** `during_prefill`/`during_decode`

## Verdict

The hook system extension is **architecturally sound** for enabling distributed inference. Main concerns are:

1. **Trait bloat** - too many methods on `PipelineHook`
2. **Dead code** - `AsyncPipelineHook`, `Layered`, `LayerExecutor` not implemented
3. **Inconsistent patterns** - error handling and stage detection

Recommended: Consolidate the trait hierarchy before more implementations are built.
