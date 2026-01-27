# Testing Guidelines for mistral.rs

This document outlines testing practices for mistral.rs, with particular attention to memory management and test organization.

## Serial Test Groups

Model loading tests can exhaust system memory when run in parallel. We use `serial_test` crate to ensure these tests run sequentially.

### Available Groups

| Group | Use Case | Example |
|-------|----------|---------|
| `#[serial(small_model)]` | Tests loading models <2B params | Qwen3-0.6B, Qwen2-0.5B |
| `#[serial(large_model)]` | Tests loading models >2B params | Phi3-3.8B, Mistral-7B |

### Cross-File Serialization

Serial groups work across test files. Tests with `#[serial(small_model)]` in `text_pipeline_integration.rs` will NOT run concurrently with tests using the same group in `safetensors_pipeline_integration.rs`.

### Example Usage

```rust
use serial_test::serial;

#[test]
#[serial(small_model)]  // Serialized with other small_model tests
fn test_model_loading() {
    let pipeline = CausalLMLoaderBuilder::from_hf_gguf(...)
        .build()
        .expect("Failed to load");
    // Test logic...
}  // Pipeline dropped here, memory released
```

## Memory Management

### Guidelines

1. **Always use serial groups for model-loading tests** - Prevents OOM from parallel loads
2. **Let pipelines drop naturally** - Rust's RAII handles cleanup
3. **Avoid holding multiple models** - Load, test, drop before loading next
4. **Use small models in CI** - Qwen3-0.6B (~400MB) is our default

### Bad Pattern

```rust
#[test]
fn test_multiple_models() {
    let model1 = load_model("repo1");  // ~2GB
    let model2 = load_model("repo2");  // ~2GB (now at 4GB!)
    compare(model1, model2);           // Memory exhaustion risk
}
```

### Good Pattern

```rust
#[test]
#[serial(small_model)]
fn test_model1() {
    let model = load_model("repo1");
    let result = model.infer("test");
    assert!(result.is_ok());
}  // model dropped, memory released

#[test]
#[serial(small_model)]
fn test_model2() {
    let model = load_model("repo2");
    let result = model.infer("test");
    assert!(result.is_ok());
}
```

## Test Organization

### Integration Tests

Located in `mistralrs-core/tests/`:

| File | Purpose |
|------|---------|
| `text_pipeline_integration.rs` | GGUF model loading via `CausalLMLoaderBuilder` |
| `safetensors_pipeline_integration.rs` | Safetensors model loading |

### Unit Tests

Inline in source files:
- `lora/registry.rs` - AdapterRegistry tests
- `lora/registry_linear.rs` - RegistryLoraLinear tests
- `attention/backends/cpu/*.rs` - CPU attention tests

### Benchmarks

Located in `mistralrs-core/benches/`:
- Run separately via `cargo bench`
- Not affected by test parallelism
- Use Criterion for statistical rigor

## Environment Variables

### Model Overrides

```bash
# Use local GGUF model instead of downloading
TEST_GGUF_MODEL=/path/to/model.gguf cargo test

# Use specific HuggingFace repo for safetensors tests
TEST_QWEN2_SAFETENSORS=my-org/qwen2 cargo test
```

### Test Control

```bash
# Skip large models (useful in CI with limited memory)
TEST_SAFETENSORS_SKIP_LARGE=1 cargo test

# Enable regression tests with expected outputs
TEST_SAFETENSORS_REGRESSION=1 cargo test

# Run LoRA adapter tests
TEST_LORA_ADAPTER=username/adapter cargo test test_lora
```

## Running Tests

### Full Test Suite

```bash
# All tests with default parallelism (serial groups respected)
cargo test

# Explicit single-threaded (safest, slowest)
cargo test -- --test-threads=1

# Just integration tests
cargo test --test text_pipeline_integration
cargo test --test safetensors_pipeline_integration
```

### Specific Test Categories

```bash
# Unit tests only (fast, no model loading)
cargo test --lib

# LoRA registry tests
cargo test lora::registry

# Attention backend tests
cargo test attention::backends
```

### Memory-Constrained Environments (CI)

```bash
# Skip large models, single thread
TEST_SAFETENSORS_SKIP_LARGE=1 cargo test -- --test-threads=1
```

## Writing New Tests

### Checklist

1. **Does it load a model?** → Add `#[serial(small_model)]` or `#[serial(large_model)]`
2. **Does it need a real model?** → Use env var override pattern
3. **Is it deterministic?** → Consider regression test with expected output
4. **Is it fast?** → Can be a unit test without serial flag

### Template for Model-Loading Tests

```rust
use serial_test::serial;

/// Test description explaining what this verifies.
#[test]
#[serial(small_model)]
fn test_feature_name() {
    // Skip if env var not set (for optional tests)
    let model_repo = match std::env::var("TEST_MY_MODEL") {
        Ok(repo) => repo,
        Err(_) => {
            println!("Skipping: TEST_MY_MODEL not set");
            return;
        }
    };

    // Load model
    let pipeline = CausalLMLoaderBuilder::from_hf_gguf(&model_repo, &["model.gguf"])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build()
        .expect("Failed to load model");

    // Test assertions
    assert!(!pipeline.name().is_empty());

    // Pipeline dropped here, memory released
}
```

## Troubleshooting

### Tests Crash with OOM

1. Check for missing `#[serial(...)]` annotations
2. Run with `--test-threads=1`
3. Use smaller models via env vars
4. Check if test holds multiple models simultaneously

### Tests Pass Locally, Fail in CI

1. CI may have less memory - skip large models
2. Network issues - use cached models
3. Parallel execution - ensure serial groups match

### Flaky Tests

1. Check for race conditions (unlikely with serial groups)
2. Verify deterministic sampling (temperature=0)
3. Check model download reliability
