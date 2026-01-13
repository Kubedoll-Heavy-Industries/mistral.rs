# Sequence State Review

**Files**: `mistralrs-core/src/sequence.rs`
**Lines Changed**: ~117

## Summary

The Sequence type represents a single inference request lifecycle. Modifications add pipeline parallelism support:

- New `request_id` field (UUID7) for distributed tracing correlation
- New methods: `set_tokens_for_pp`, `append_tokens_for_pp`, `set_responder`, `set_prompt_len`
- Simplified `len()` method (now returns `self.tokens.len()` directly)
- Chunked prefill helpers: `is_final_prefill_chunk`, `sequence_position`
- Removed `pipeline_continue_op_id` (consolidated into `request_id`)

## Strengths

### 1. Clean UUID7 Integration
- Private field with public getter
- Properly threaded through constructor
- UUID7 provides time-ordering for distributed tracing

### 2. Good Encapsulation of PP Methods
```rust
/// Append raw tokens for pipeline parallelism continuation.
/// Unlike `add_token()`, this doesn't update logprobs or completion bytes -
/// it just tracks the token for position/length calculation.
pub fn append_tokens_for_pp(&mut self, tokens: &[u32])
```

### 3. Proper Block Metadata Documentation
```rust
/// IMPORTANT: Does NOT modify block metadata (paged attention state) because
/// KV cache grows incrementally via forward passes.
```

### 4. Consolidated Request ID
Removing `pipeline_continue_op_id` in favor of `request_id` reduces redundancy.

## Issues

### Issue 1: ~~`len()` Semantic Change~~ ✅ ANALYZED & FIXED
**Location**: `sequence.rs:701-707`, `inputs_processor.rs:447-454`
**Severity**: ~~Medium~~ Resolved

**Change**: `len()` now returns `tokens.len()` instead of KV cache dimensions.

**Analysis**: This is part of a coordinated design for chunked prefill:
- `token_offset` tracks absolute position
- `prefill_chunk_offset` tracks position within chunked prefill
- `prompt_len` (set via `set_prompt_len()`) tracks full prompt length for PP

**Call site safety:**
| Site | Usage | Safe? |
|------|-------|-------|
| RoPE (decode) | `max(token_offset, len()-1)` | ✅ Uses max() |
| RoPE (prefill) | `token_offset + prefill_chunk_offset` | ✅ Explicit |
| Sampling | `len() == prompt_len` | ✅ Only at Stage 0 |
| Paged attn context | `seq.len()` directly | ✅ **FIXED** |

**Fix Applied**: Changed paged attention context calculation from `seq.len()` to `seq.token_offset() + seq.len()`:
```rust
// Total context = token_offset (prefix/PP position) + current buffer length
let total_context = seq.token_offset() + seq.len();
```

This correctly handles: normal (0+N=N), prefix cache (cached+rest=total), PP (pos+1=total).

### Issue 2: ~~HIGH - Debug Logging in Hot Path~~ ✅ FIXED
**Location**: `sequence.rs:777-786`
**Severity**: ~~High~~ Resolved

Removed PP DEBUG logging block that ran on every forward pass.

### Issue 3: ~~Dead Code~~ ✅ FIXED
**Location**: `sequence.rs:767-769`
**Severity**: ~~Low~~ Resolved

Removed `let _ = chunk_size;` dead code.

### Issue 4: No State Transition Validation
**Location**: `sequence.rs:1079-1085`
**Severity**: Medium

```rust
pub fn set_state(&self, state: SequenceState) {
    // No validation of valid transitions
    *self.state.write().unwrap() = state;
}
```

Invalid transitions possible:
- `Waiting -> Done` (skipping prompt/completion)
- `Done -> RunningPrompt` (restarting finished)
- `Error -> RunningCompletion` (invalid recovery)

**Recommendation**: Add debug assertions for valid transitions.

### Issue 5: `set_prompt_len` Allows Inconsistent State
**Location**: `sequence.rs:805-809`
**Severity**: Medium

```rust
pub fn set_prompt_len(&mut self, len: usize) {
    self.prompt_len = len;  // Can be > tokens.len()
}
```

**Recommendation**: Add assertion:
```rust
debug_assert!(len <= self.tokens.len() || self.tokens.is_empty());
```

### Issue 6: Inconsistent UUID Handling Across Codebase
**Location**: Multiple files
**Severity**: Low

- Server-core: `state.next_request_id()` -> UUID7
- Library crate: `uuid::Uuid::nil()` for all requests
- Interactive mode: Mixed

**Recommendation**: Document that `nil()` is valid for local-only inference.

### Issue 7: `set_responder` Safety
**Location**: `sequence.rs:1069-1073`
**Severity**: Low

Replacing responder mid-sequence could send to wrong receiver.

**Recommendation**: Document when this is safe (activation boundaries only).

### Issue 8: `append_tokens_for_pp` vs `set_tokens_for_pp` Asymmetry
**Location**: `sequence.rs:997-1005`
**Severity**: Low

One updates block metadata, the other doesn't. Could lead to subtle bugs.

**Recommendation**: Consider renaming for clarity:
- `append_tokens_for_pp` -> `append_tokens_with_blocks`
- `set_tokens_for_pp` -> `replace_tokens_only`

## Recommendations Summary

1. **Remove debug logging** in `get_toks()` before merge
2. **Remove dead code** (`let _ = chunk_size;`)
3. **Test `len()` change** thoroughly with xlora and cached inference
4. **Consider state transition validation** for PP scenarios
5. **Add safety documentation** to `set_responder` and `set_prompt_len`
6. **Standardize UUID handling** across server and library APIs

## Verdict

The sequence changes enable PP but have a **critical semantic change** in `len()` that needs verification. Debug logging must be removed.
