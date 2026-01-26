# Model Migration Plan: transformer_builder Infrastructure

## Overview

Migrate remaining quantized models to use the `transformer_builder` infrastructure established in `quantized_qwen3.rs`. This reduces code duplication and ensures consistent loading patterns across all models.

**Reference Implementation:** `mistralrs-core/src/models/quantized_qwen3.rs` (248 lines)

**Target Reduction:** Approximately 50% code reduction per model (based on Qwen3 migration)

---

## Infrastructure Components

Located in `mistralrs-core/src/pipeline/loaders/transformer_builder.rs`:

### Existing Infrastructure

| Component | Purpose |
|-----------|---------|
| `TransformerConfig` | Common transformer hyperparameters |
| `TransformerConfig::from_gguf_metadata()` | Parse config from GGUF |
| `WeightSource` trait | Abstraction over weight formats |
| `GgufWeightSource<'a, 'c, R>` | GGUF implementation |
| `TensorNaming` trait | Tensor name patterns |
| `GgufNaming` | Standard GGUF naming (blk.N.attn_q.weight) |
| `StandardTransformerBlock` | Pre-built attention + MLP block |
| `load_transformer_layers()` | Generic layer loading with customizer |

### Infrastructure Extensions Needed

Some models require features not yet in transformer_builder:

| Feature | Needed By | Status |
|---------|-----------|--------|
| LayerNorm (not RmsNorm) | Phi2, Starcoder2 | ✅ Implemented (`load_layer_norm`) |
| Attention biases | Qwen2, Phi2, Starcoder2 | ✅ Implemented (`load_linear_with_optional_bias`, `with_attention_bias`) |
| Non-gated MLP | Phi2, Starcoder2 | ⚠️ Exists (`NonGatedMlp`) but not in builder |
| Fused QKV projection | Phi3 | Not implemented |
| Custom RoPE (YaRN) | Mistral3 | ✅ Implemented (`impl PositionEncoding for Mistral3RotaryEmbedding`) |
| Custom RoPE (Phi3) | Phi3 | Not implemented |

---

## Model Migration Priority

### Tier 1: Direct Migration (No Infrastructure Changes)

These models can be migrated immediately using existing infrastructure:

#### 1. `quantized_qwen.rs` (Qwen2) - ✅ MIGRATED

**Original:** 566 lines → **Result:** 249 lines (56% reduction)

**Migration completed:**
- Uses `TransformerConfig::from_gguf_metadata().with_attention_bias()`
- Uses `load_transformer_layers()` with customizer for optional Q/K norm
- Supports both "qwen2" and "qwen3" architectures

---

### Tier 2: Infrastructure Extension Required

These models need transformer_builder extensions first:

#### 2. `quantized_starcoder2.rs` - 339 lines

**Architecture:** LayerNorm, non-gated MLP, biases throughout

**Required Extensions:**
- `WeightSource::load_layer_norm()` method
- `TransformerLayerBuilder::non_gated_mlp()` option
- Bias support in attention projections

**Migration Strategy:**
1. Add LayerNorm support to WeightSource
2. Add non-gated MLP variant to StandardTransformerBlock
3. Migrate model using extended infrastructure

**Estimated Result:** ~80 lines (76% reduction)

#### 3. `quantized_phi2.rs` - 485 lines

**Architecture:** Similar to Starcoder2 (LayerNorm, non-gated MLP, biases)

**Required Extensions:** Same as Starcoder2

**Migration Strategy:**
1. Reuse Starcoder2 infrastructure extensions
2. Phi2-specific: partial rotary embeddings handling

**Estimated Result:** ~90 lines (81% reduction)

#### 4. `quantized_phi3.rs` - 588 lines

**Architecture:** Fused QKV, custom rope with su/long factors

**Required Extensions:**
- Fused QKV projection loading
- Phi3-specific rotary embedding type

**Migration Strategy:**
1. Add fused QKV loading to WeightSource
2. Keep Phi3RotaryEmbedding as model-specific code
3. Customizer creates attention with fused QKV

**Estimated Result:** ~120 lines (80% reduction)

#### 5. `quantized_mistral3.rs` - ✅ MIGRATED

**Original:** 1387 lines → **Result:** 1202 lines (13% reduction)

**Architecture:** YaRN RoPE scaling, otherwise standard Llama-like

**Migration completed:**
- Added `impl PositionEncoding for Mistral3RotaryEmbedding`
- Uses `TransformerConfig::from_gguf_metadata()` for base config
- Uses `load_transformer_layers()` with customizer to inject YaRN RoPE
- Kept `YarnConfig`, `Mistral3RotaryEmbedding`, and extensive tests
- Simplified `from_gguf` and `run_layers` methods

**Note:** Most of the file is YaRN-specific code (~290 lines) and tests (~550 lines)
that must remain. The `from_gguf` method was reduced from ~230 to ~150 lines.

---

### Tier 3: MoE Models (Separate Pattern)

#### 6. `quantized_qwen3_moe.rs` - 644 lines

**Architecture:** MoE with expert routing

**Note:** MoE models follow a different pattern (see `mixtral.rs`). Consider whether to:
- Create `MoeTransformerBlock` infrastructure
- Keep as separate implementation

**Decision:** Defer until dense model migrations complete

---

### Already Migrated / Not Migrating

| Model | Status | Notes |
|-------|--------|-------|
| `quantized_qwen3.rs` | ✅ Done | Reference implementation |
| `quantized_qwen.rs` | ✅ Done | Migrated with attention bias support (566→249 lines) |
| `quantized_mistral3.rs` | ✅ Done | YaRN RoPE injected via customizer (1387→1202 lines) |
| `quantized_llama.rs` | Skip | Safetensors primary, has both paths |
| `mixtral.rs` | ✅ Done | New MoE implementation |

---

## Parallel Work Packages

### Package A: Qwen2 Migration - ✅ COMPLETE

**Status:** Migrated successfully (566 → 249 lines, 56% reduction)

**Commits:**
- `76d9a3d1a refactor(models): extend transformer_builder and migrate Qwen2`

**Implementation:**
- Uses `TransformerConfig::from_gguf_metadata().with_attention_bias()`
- Uses `load_transformer_layers()` with customizer for optional Q/K norm
- Supports both "qwen2" and "qwen3" architectures via the same file

### Package B: LayerNorm + Non-gated MLP Infrastructure - ⏳ PARTIAL

**Completed:**
- ✅ `WeightSource::load_layer_norm()` - Loads LayerNorm (weight + bias)
- ✅ `WeightSource::load_linear_with_optional_bias()` - Loads linear with optional bias
- ✅ `TransformerConfig::with_attention_bias()` - Enable Q/K/V bias loading

**Remaining:**
- ❌ Non-gated MLP support in `TransformerLayerBuilder` (need generic return type)
- ❌ Alternative block type for LayerNorm models (current `StandardTransformerBlock` uses RmsNorm)

**Commits:**
- `76d9a3d1a refactor(models): extend transformer_builder and migrate Qwen2`

**Note:** The `NonGatedMlp` struct exists in `layers.rs` and implements `FeedForward`.
To use it with transformer_builder, need to make `load_transformer_layers` generic
over the block type or create a separate loading function.

### Package C: Starcoder2 Migration

**Depends on:** Package B

**Agent Task:**
```
Migrate quantized_starcoder2.rs using new LayerNorm infrastructure.

Uses:
- LayerNorm (not RmsNorm)
- Non-gated MLP
- Attention biases

Follow quantized_qwen3.rs pattern with:
- TransformerConfig::from_gguf_metadata()
- load_transformer_layers() with customizer for biases
- Use with_layer_norm() and with_non_gated_mlp()

Test: cargo test -p mistralrs-core --lib
```

### Package D: Phi2 Migration

**Depends on:** Package B

**Agent Task:**
```
Migrate quantized_phi2.rs using LayerNorm infrastructure.

Similar to Starcoder2 but with:
- Partial rotary embeddings (rope applies to subset of head_dim)
- Different tensor naming patterns

Handle partial rotary via customizer or separate rotary type.

Test: cargo test -p mistralrs-core --lib
```

### Package E: Phi3 Migration

**Agent Task:**
```
Migrate quantized_phi3.rs to transformer_builder.

Phi3-specific handling:
1. Keep Phi3RotaryEmbedding (su/long factors) - model-specific
2. Add fused QKV support:
   - Load single qkv_proj tensor
   - Split into q, k, v in customizer or attention forward

Note: Phi3 has complex rope with short/long scaling factors.
Keep that logic but simplify layer loading.

Test: cargo test -p mistralrs-core --lib
```

### Package F: Mistral3 Migration - ✅ COMPLETE

**Status:** Migrated successfully (1387 → 1202 lines, 13% reduction)

**Implementation:**
- Added `impl PositionEncoding for Mistral3RotaryEmbedding`
- Uses `TransformerConfig::from_gguf_metadata()` for base config
- Uses `load_transformer_layers()` with customizer to inject YaRN RoPE
- Kept `YarnConfig`, `Mistral3RotaryEmbedding`, and extensive tests (~550 lines)
- Simplified `from_gguf` method to use generic layer loading

**Note:** Modest line reduction due to extensive YaRN RoPE implementation and tests
that are essential to keep. The key benefit is consistency with other models.

---

## Verification Checklist

After each migration:

1. **Compilation:** `cargo check -p mistralrs-core`
2. **Linting:** `cargo clippy -p mistralrs-core -- -D warnings`
3. **Unit tests:** `cargo test -p mistralrs-core --lib`
4. **Integration tests:** `cargo test -p mistralrs-core --test text_pipeline_integration`
5. **Model loading:** Set `TEST_<MODEL>_MODEL=/path/to/model.gguf` and run family test

---

## Infrastructure Files Reference

```
mistralrs-core/src/
├── models/
│   ├── mod.rs                    # Model traits (Model, TransformerModel, LanguageModel)
│   ├── quantized_qwen3.rs        # Reference implementation (ALREADY MIGRATED)
│   ├── quantized_qwen.rs         # Package A
│   ├── quantized_starcoder2.rs   # Package C
│   ├── quantized_phi2.rs         # Package D
│   ├── quantized_phi3.rs         # Package E
│   └── quantized_mistral3.rs     # Package F
└── pipeline/loaders/
    ├── transformer_builder.rs    # Infrastructure (Package B extends this)
    └── mod.rs                    # Exports
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Total lines (6 models) | From ~4400 to ~750 (~83% reduction) |
| All tests pass | 100% |
| No new clippy warnings | 0 warnings |
| Model family e2e tests | All pass with model files |
