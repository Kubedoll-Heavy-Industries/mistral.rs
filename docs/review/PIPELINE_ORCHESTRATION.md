# Pipeline Orchestration Review

**Files**: `pipeline/mod.rs`, `pipeline/gguf.rs`, `pipeline/normal.rs`, `pipeline/inputs_processor.rs`
**Lines Changed**: ~1004

## Summary

The pipeline module has been refactored to support pipeline parallelism with chunked prefill:

1. **`InferenceStep` enum** (`inputs_processor.rs:631-708`): State machine for Prefill vs Decode
2. **Extracted `forward_pass()` method** (`mod.rs:514-614`): Consolidates forward logic for PP
3. **New `handle_result()` method** (`mod.rs:720-776`): Extracted result dispatch
4. **Refactored `step()` method** (`mod.rs:619-716`): Simplified from ~250 to ~90 lines
5. **PP coordination in GGUF/Normal pipelines**: Stage-level orchestration

## Strengths

### 1. Clean State Machine Design (`InferenceStep`)
```rust
pub enum InferenceStep {
    Prefill {
        chunk_tokens: usize,
        chunk_start_position: usize,
        total_prompt_tokens: usize,
        is_final_chunk: bool,
    },
    Decode {
        current_position: usize,
    },
}
```

Clear discrimination, structured metadata, good helper methods (`is_final_chunk()`, `is_decode()`, `current_position()`).

### 2. Massive `step()` Complexity Reduction
From ~250 lines with nested loops to ~90 lines. Old code had:
- Duplicated iteration logic
- Nested `for` loops with index tracking
- Duplicated result dispatch

### 3. Proper Hook Abstraction
`HookContainer` provides stage-level operations:
- `is_first_stage()`, `is_last_stage()`
- `send_stage_output()`, `receive_stage_input()`

### 4. Request Context Threading
`request_id` and `inference_step` added to `ModelInputs` for proper correlation.

## Issues

### Issue 1: CRITICAL - Duplicate `init_pipeline_request` Calls
**Location**: `gguf.rs:852-869` and `gguf.rs:910-914`
**Severity**: Critical

```rust
// First call at top of forward_inputs (lines 856-862):
if let InferenceStep::Prefill { total_prompt_tokens, .. } = inference_step {
    hook.call_init_pipeline_request(request_id, total_prompt_tokens);
}

// Second call inside Model::Phi3 match arm (lines 910-914):
if let Some(hook) = self.model.get_hook() {
    if let InferenceStep::Prefill { total_prompt_tokens, .. } = inference_step {
        hook.call_init_pipeline_request(request_id, total_prompt_tokens);
    }
}
```

**Results in duplicate RPC calls** for Phi3 models during prefill.

**Recommendation**: Remove the second call (lines 910-914).

### Issue 2: HIGH - Silent Fallback on PP Receive
**Location**: `normal.rs:1237-1243`
**Severity**: High

```rust
None => {
    tracing::warn!(
        "Middle/last stage didn't receive activation, using local embeddings"
    );
    self.model.get_input_embeddings(&input_ids)?  // PRODUCES GARBAGE
}
```

If middle/last stage fails to receive activation, it silently falls back to local embeddings. This produces **incorrect output** while appearing to work.

**Recommendation**: Error instead of fallback:
```rust
None => {
    return Err(candle_core::Error::Msg(
        "Middle/last stage must receive activation from previous stage".to_string(),
    ));
}
```

### Issue 3: HIGH - Inconsistent PP Handling Between Pipelines
**Location**: `gguf.rs` vs `normal.rs`
**Severity**: High

- **GGUF Qwen3/Phi3**: PP logic inside match arms per-model (copy-pasted)
- **Normal pipeline**: PP logic in single `if let Some(ref hook)` block

When PP behavior changes, both must be updated differently.

**Recommendation**: Extract PP logic into shared helper:
```rust
fn execute_pipeline_stage(
    hook: &HookContainer,
    get_embeddings: impl FnOnce() -> Result<Tensor>,
    run_layers: impl FnOnce(Tensor) -> Result<Tensor>,
    get_logits: impl FnOnce(Tensor) -> Result<Tensor>,
    tokens: &[u32],
    request_id: Uuid,
    sequence_position: usize,
) -> Result<Tensor>
```

### Issue 4: MEDIUM - `forward_pass()` Only Handles CausalGeneration
**Location**: `mod.rs:569-572`
**Severity**: Medium

```rust
match result {
    ForwardInputsResult::CausalGeneration { logits } => Ok(logits),
    _ => candle_core::bail!("forward_pass expects CausalGeneration result"),
}
```

If called with embeddings/raw logits model, errors. Fine for PP-only use but limits reusability.

### Issue 5: MEDIUM - Unused `mut` Binding
**Location**: `normal.rs:1216`
**Severity**: Medium

```rust
let mut logits = match self.model.is_xlora() {  // mut never used
```

### Issue 6: MEDIUM - Hardcoded Prefill Detection
**Location**: `normal.rs:1272`
**Severity**: Medium

```rust
let is_decode = tokens.len() == 1;
```

Uses token count instead of available `inference_step` state machine.

**Recommendation**: Use `inference_step.is_decode()`.

### Issue 7: LOW - Unwrap Chains in `handle_result`
**Location**: `mod.rs:731-744`
**Severity**: Low

```rust
.map(|idx| vec![logits.i(idx).unwrap().to_device(&Device::Cpu).unwrap()])
```

Should use `?` for error propagation.

### Issue 8: LOW - Tracing Noise
**Location**: Multiple in `gguf.rs` and `normal.rs`
**Severity**: Low

Extensive `tracing::info!` for normal PP flow. Should be `debug!`.

## Recommendations Summary

1. **Remove duplicate `init_pipeline_request`** in GGUF Phi3
2. **Error on missing activation** instead of silent fallback
3. **Extract PP logic** into shared helper function
4. **Use `InferenceStep`** in NormalPipeline instead of `tokens.len() == 1`
5. **Demote PP tracing** to debug level
6. **Propagate errors** with `?` instead of `.unwrap()`

## Verdict

The refactoring achieves its goal of cleaning up pipeline orchestration. The `InferenceStep` state machine and `step()` simplification are well-done.

However, there are **critical correctness issues**:
1. Duplicate init calls waste RPC
2. Silent fallback produces garbage output
3. Divergent PP patterns create maintenance burden

These must be fixed before merge.
