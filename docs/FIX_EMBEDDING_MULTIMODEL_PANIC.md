# Fix: Embedding Model Panic in Multi-Model Mode

## Problem Summary

When running embedding models alongside text models in multi-model mode, the embedding engine panics at `mistralrs-core/src/sequence.rs:523` with "entered unreachable code".

**Root Cause**: In multi-model mode, the `scheduler_config` from the first model is reused for all subsequent models (line 932 in `mistralrs_for_server_builder.rs`). If the first model has PagedAttention enabled but an embedding model doesn't support it, the embedding model still gets a PagedAttention scheduler, which calls methods that panic on non-PagedAttention sequences.

## Analysis: Why Option Types Are NOT Recommended

Initially we considered changing `BlockEngineSequence` trait methods to return `Option` types. After analysis, this is **not recommended** because:

1. **Defeats the purpose**: All 7+ call sites would need `.expect()`, making `Option` pointless
2. **Risk of silent failures**: Using `.unwrap_or(0)` could hide bugs instead of failing fast
3. **High coupling**: All call sites assume direct values - changing breaks compilation at 6+ locations
4. **Doesn't fix root cause**: The real issue is scheduler config reuse, not the trait API

The `unreachable!()` panics are actually correct behavior - if PagedAttention code is called on non-PagedAttention sequences, that's a bug that should fail loudly.

## Fix Strategy

**Single targeted fix**: Compute per-model scheduler config in multi-model mode.

This ensures:
- Text models with PagedAttention get `SchedulerConfig::PagedAttentionMeta`
- Embedding/rerank/speech models get `SchedulerConfig::DefaultScheduler`
- Each model type uses appropriate scheduler, no cross-contamination

---

## Implementation Plan

### Step 1: Fix Multi-Model Scheduler Config

**File**: `mistralrs-server-core/src/mistralrs_for_server_builder.rs`

**Location**: Lines ~928-936 in the additional models loop

**Current code** (problematic):
```rust
mistralrs
    .add_model(
        pipeline_name.clone(),
        pipeline,
        scheduler_config.clone(),  // <- REUSES first model's config!
        add_model_config,
    )
```

**Fixed code**:
```rust
// Compute scheduler config for THIS specific model
let model_scheduler_config = init_scheduler_config(
    &cache_config,
    &pipeline,
    self.max_seqs,
    self.max_prefill_chunk_size,
).await;

mistralrs
    .add_model(
        pipeline_name.clone(),
        pipeline,
        model_scheduler_config,  // <- Model-specific config
        add_model_config,
    )
```

### Why This Works

The `init_scheduler_config()` function (lines 1133-1158) checks `pipeline.get_metadata().cache_config`:
- If `cache_config` is `Some(...)` → returns `SchedulerConfig::PagedAttentionMeta`
- If `cache_config` is `None` → returns `SchedulerConfig::DefaultScheduler`

Embedding models set `cache_config: None` during loading (see `embedding.rs:618`), so they will correctly receive `DefaultScheduler`.

---

## GPU Memory Consideration

**Note**: Each model's PagedAttention scheduler allocates its own KV cache blocks. When running multiple PagedAttention models on the same GPU:

- Current behavior: First model's cache config is reused (incorrect but avoids double-allocation)
- Fixed behavior: Each model computes its own cache config

**Potential issue**: If both models request large KV caches, total GPU memory usage increases.

**Mitigation options** (out of scope for this fix, but document for future):
1. Scale `paged_attn_gpu_mem` by number of PagedAttention models
2. Add per-model memory configuration in multi-model config
3. Implement shared block engine across models (major refactor)

For now, embedding models don't use PagedAttention, so this fix doesn't introduce memory issues for the embedding + text model use case.

---

## Files to Modify

| File | Changes |
|------|---------|
| `mistralrs-server-core/src/mistralrs_for_server_builder.rs` | Call `init_scheduler_config()` per model (~line 932) |

**Total**: 1 file, ~5 lines changed

---

## Verification

1. **Compile check**: `cargo check --workspace`
2. **Run tests**: `cargo test -p mistralrs-core -p mistralrs-server-core`
3. **Clippy**: `cargo clippy --workspace --tests`
4. **Manual test**: Run multi-model config with embedding model:
   ```bash
   ./target/debug/mistralrs-server --port 8081 multi-model \
     --config examples/qwen-multi-model-config.json \
     --default-model-id Qwen/Qwen3-4B
   ```
5. **Test embedding endpoint**:
   ```bash
   curl -X POST http://localhost:8081/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["test"]}'
   ```
6. **Test text model still works**:
   ```bash
   curl -X POST http://localhost:8081/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

---

## Future Improvements (Out of Scope)

1. **Add integration test**: Multi-model mode with mixed model types (text + embedding)
2. **Memory management**: Shared block engine or per-model memory budgets for multi-model PagedAttention
3. **Better error messages**: If `unreachable!()` still triggers somehow, improve panic message with context
