# Pipeline Refactoring Plan

## Goal

Align the pipeline implementation with the architecture described in `architecture.md`. The core insight: **models are stateless ML primitives; pipelines own runtime state and orchestrate execution**.

See also:
- **[architecture.md](./architecture.md)** — Core type hierarchy and design principles
- **[type-driven-design.md](./type-driven-design.md)** — FP/category theory foundations

---

## Current State (2025-01-25)

### Phase 1: Model Migration ✅ COMPLETE

All GGUF models now implement `TransformerModel`:
- ✅ Phi3, Starcoder2, Qwen3 (initial)
- ✅ Llama, Qwen (session 2)
- ✅ Mistral3, Phi2, Qwen3MoE (session 3)
- ⏸️ XLoraLlama, XLoraPhi3 (deferred — different forward signature)

### Phase 2: Model Enum Dissolution ✅ COMPLETE

```rust
// Before: 10-variant enum with match everywhere
enum Model { Llama(QLlama), Phi3(QPhi3), ... }

// After: 3-variant enum (Transformer + 2 XLora)
enum ModelVariant {
    Transformer(Box<dyn TransformerModel + Send + Sync>),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
}
```

### Phase 3: Loaders Module (COMPLETE)

**Vision**: Clean interface for opening model files and extracting components.

These aren't "weight sources" — they're complete **Loaders**. GGUF contains weights + tokenizer + config + chat template.

**Implementation** (in `src/loaders/`):
- ✅ `GgufLoader` — opens GGUF files, extracts metadata/tokenizer/chat template
- ✅ `GgufMetadata` — architecture, num_layers, hidden_size, rope config, etc.
- ✅ `LoaderMetadata` trait — format-specific metadata via downcasting
- ⏸️ `SafetensorsLoader` — future work when needed

**Key design decisions:**
1. **No model construction in loader** — Model construction stays with `FromGGUF` trait
2. **No layer sharding in loader** — That's a construction concern (`ParamsGGUF` already has `Option<Range<usize>>`)
3. **Model-specific config parsing** — Each model has its own `PropsGGUF` struct with `TryFrom<ContentMetadata>`

```rust
// GgufLoader provides file access
let loader = GgufLoader::open(&["model.gguf"])?;
let arch = loader.architecture();
let tokenizer = loader.tokenizer()?;

// Model construction uses existing FromGGUF machinery
// Each model parses its own config via ContentMetadata
let metadata = ContentMetadata {
    path_prefix: "mistral3",
    metadata: content.get_metadata(),
};
let props = PropsGGUF::try_from(metadata)?;  // Model-specific parsing
```

| Loader | Provides |
|--------|----------|
| `GgufLoader` | Open files, architecture, tokenizer, chat template, metadata |
| `FromGGUF` (existing) | Model construction with `ContentMetadata` for config parsing |

### Phase 4: Unified Pipeline (IN PROGRESS)

See **[architecture-loader-migration.md](./architecture-loader-migration.md)** for the full plan.

**Simplified approach**: No type parameters on loaders. Architecture dispatch via `GGUFArchitecture` enum match.

```rust
// Loader extracts metadata (no type parameter)
let loader = GgufLoader::open(&["model.gguf"])?;
let arch = loader.gguf_architecture();
let tokenizer = loader.tokenizer()?;

// Model construction via enum dispatch
let model: Box<dyn TransformerModel> = match arch {
    GGUFArchitecture::Qwen3 => Box::new(QQwen3::from_gguf(...)?),
    GGUFArchitecture::Llama => Box::new(QLlama::from_gguf(...)?),
    // ...
};

// Unified pipeline works with any TransformerModel
let pipeline = TextPipeline::new(model, tokenizer, ...);
```

**Key insight**: A model's implementation, architecture, and family all refer to the same thing. Don't over-abstract.

### Key Design Patterns Established

1. **Stateless Models** — Pipeline owns cache, models are pure functions
2. **TransformerBlock<N, A, F>** — Generic composition, zero-cost abstraction
3. **Builder Pattern** — `CausalAttention::new().with_qk_norm().with_paged_attn()`
4. **Layer-Range PP** — Models don't know about PP; `has_embedding`/`has_lm_head` determines calls
5. **Trait Hierarchy** — `TransformerModel: Model` (capability-based, not identity-based)

### Anti-Patterns to Remove

1. **Mixin Traits** — Python pattern, doesn't fit Rust (see Remaining #1)
2. **Forced Supertraits** — Pipelines shouldn't require capabilities they don't use

---

## What's Fixed vs What Remains

### Fixed
- Model enum explosion (10 variants → 3)
- Stateful models (cache moved to pipeline)
- PP as special case (now layer-range driven)
- Device access fragmentation (trait method)

### Remaining

#### 1. Mixin Anti-Pattern (PRIORITY)
The "mixin" terminology and design are **cargo-culted from Python** and don't fit Rust:

```rust
// ANTI-PATTERN: Forced supertraits
pub trait Pipeline:
    + PreProcessingMixin      // Unused supertrait dep on MetadataMixin
    + IsqPipelineMixin        // GGUF just returns error
    + CacheManagerMixin       // Actually useful
    + MetadataMixin           // 6 unrelated methods dumped together
    + AnyMoePipelineMixin     // Mostly unreachable!() stubs
```

**Problems:**
- Forces all pipelines to implement irrelevant traits
- `MetadataMixin` violates SRP (device, tokenizer, name, XLora state, metadata, mapper)
- `AnyMoePipelineMixin` is stubs for non-MoE pipelines
- `PreProcessingMixin: MetadataMixin` creates hidden coupling

**Solution:** Replace with optional capability traits (see `type-driven-design.md`):
```rust
trait Pipeline { fn forward(...); fn device(); fn name(); }
trait Tokenizing: Pipeline { fn tokenizer(); }
trait ChatCapable: Tokenizing { fn chat_template(); }
// Callers request capabilities via bounds, not forced inheritance
```

#### 2. XLora Forward Signature
XLora models have different forward signature. Options:
- Separate `XLoraModel` trait
- XLora-specific pipeline
- Context parameter with XLora state

#### 3. Loader/Pipeline Conflation
GGUF, Safetensors, GGML are currently "pipelines" but should be "loaders". The same `UnifiedPipeline` should work regardless of weight format.

## Target Architecture

From `architecture.md`:

```
Pipeline (trait)
  ├── Model (weights + config, stateless)
  ├── Cache (KV cache, mutable runtime state)
  ├── Hooks (PP streaming, callbacks)
  └── Device placement
```

### Pipeline Should Own
- KV cache (✓ already moved for migrated models)
- Hooks for PP/callbacks (✓ done)
- Device placement info
- Tokenizer
- Chat template

### Model Should Provide
- `embed()`, `transform()`, `lm_head()` (stateless)
- `device()` — where weights live
- `num_layers()`, `max_seq_len()` — config queries
- Weights (immutable after load)

## Refactoring Steps

### Phase 1: Quick Wins (No Trait Changes)
1. Remove redundant `max_seq_len` match in loader
2. Add `Model::device()` method to consolidate MetadataMixin
3. Simplify `Model::num_layers()` for TransformerModel models

### Phase 2: Expand TransformerModel
1. Add `device() -> &Device` to TransformerModel trait
2. Add `max_seq_len() -> usize` to TransformerModel trait
3. `Model` methods delegate to trait for migrated models

### Phase 3: Migrate Remaining Models
Priority order:
1. `Qwen` — similar to Qwen3
2. `Llama` — most common
3. `Mistral3`
4. `Phi2`
5. `Qwen3MoE` — needs MoE support in trait
6. `XLoraLlama`, `XLoraPhi3` — different forward signature

### Phase 4: Dissolve Model Enum
Once all models implement `TransformerModel`:
```rust
struct GGUFPipeline {
    model: Box<dyn TransformerModel + Send + Sync>,
    // ...
}
```

### Phase 5: Consolidate Mixins
Examine what each mixin provides and whether it belongs on:
- The `Pipeline` trait itself
- A `Model` trait
- Configuration structs

## Open Questions

1. Should `TransformerModel` be object-safe? (Currently yes via `&dyn TransformerModel`)
2. How to handle XLora's different forward signature?
3. Should there be a `ModelConfig` trait for `device()`, `num_layers()`, etc.?
4. How do vision/multimodal models fit?

## Files to Examine

- `mistralrs-core/src/pipeline/mod.rs` — trait definitions, mixins
- `mistralrs-core/src/pipeline/gguf.rs` — GGUF pipeline implementation
- `mistralrs-core/src/pipeline/safetensors.rs` — Safetensors pipeline
- `mistralrs-core/src/models/mod.rs` — TransformerModel trait
