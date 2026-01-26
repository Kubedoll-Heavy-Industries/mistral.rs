# Loader Module Design

## Overview

This document captures the design for restructuring model loading in mistral.rs to cleanly separate:
1. **Models** - Stateless ML primitives (weights + forward pass)
2. **Loaders** - Deserialization (GGUF, Safetensors, ONNX) → Model
3. **Pipelines** - Runtime orchestration (cache, sampling, hooks)

Key insight: GGUF vs Safetensors are **loaders**, not pipelines. The same pipeline type can load from either format.

---

## Current State (2025-01-24)

### Completed
- All GGUF models implement `TransformerModel` trait (stateless)
- Pipeline owns KV cache for GGUF models
- Existing `FromGGUF` trait in `utils/model_config.rs`

### Remaining
- `Model` enum in `gguf.rs` should become `Box<dyn TransformerModel>`
- Safetensors models still own cache (`NormalModel` trait)
- No `FromSafetensors` trait exists
- XLora models need separate trait (`XLoraModel`)

---

## Module Structure

```
mistralrs-core/src/loaders/
├── mod.rs              # LoadContext, traits, re-exports
├── gguf.rs             # GGUFLoader + FromGGUF trait
├── safetensors.rs      # SafetensorsLoader + FromSafetensors trait
└── config.rs           # Common transformer config types
```

---

## Core Types

### LoadContext

Consolidates loading parameters shared across all formats:

```rust
pub struct LoadContext {
    pub device: Device,
    pub dtype: DType,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub attention_impl: AttentionImplementation,
    pub layer_range: Option<Range<usize>>,  // Pipeline parallelism
    pub lora_config: Option<LoraLoadConfig>, // Optional LoRA adapters
}

impl LoadContext {
    pub fn is_first_stage(&self) -> bool { ... }
    pub fn is_last_stage(&self, total_layers: usize) -> bool { ... }
}
```

### LoRA Configuration

```rust
pub struct LoraLoadConfig {
    pub adapter_paths: Vec<PathBuf>,
    pub ordering: Ordering,
    pub configs: Vec<LoraConfig>,
}

pub struct XLoraLoadConfig {
    pub lora: LoraLoadConfig,
    pub xlora_config: XLoraConfig,
    pub classifier_path: PathBuf,
}
```

---

## Model Trait Hierarchy

### Base Traits

```rust
/// Base trait for all models - weights on a device
pub trait Model: Send + Sync {
    fn device(&self) -> &Device;
}

/// Transformer-specific operations (stateless)
pub trait TransformerModel: Model {
    fn num_layers(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>;
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}
```

### XLoRA Trait (Separate)

XLoRA requires two-pass inference with dynamic adapter mixing:

```rust
/// Models with dynamic adapter mixing (XLoRA).
/// Requires two-pass inference: scaling pass → real pass.
pub trait XLoraModel: Model {
    fn get_classifier(&self) -> &XLoraClassifier;

    /// Run scaling pass to compute adapter mixing weights
    fn compute_scalings(
        &self,
        input_ids: &Tensor,
        cache: &mut [KvCache],
    ) -> Result<Tensor>;

    /// Run forward with pre-computed scalings
    fn forward_with_scalings(
        &self,
        input_ids: &Tensor,
        scalings: &Tensor,
        cache: &mut [KvCache],
    ) -> Result<Tensor>;
}
```

### Why Separate Traits?

| Aspect | Base/LoRA Model | XLoRA Model |
|--------|-----------------|-------------|
| Linear layers | `QLinear` / `QLoraLinear` | `QLoraLinear` |
| Extra components | None | Classifier |
| Forward signature | `(tokens, cache)` | `(tokens, scalings, is_scaling_pass, cache)` |
| Inference passes | 1 | 2 (scaling + real) |
| Compatible with `TransformerModel`? | Yes | **No** |

LoRA models use `QLoraLinear` layers but have the same forward interface as base models.
XLoRA fundamentally changes the inference pattern.

---

## Loader Traits

### FromGGUF

```rust
pub trait FromGGUF: Sized + TransformerModel {
    fn from_gguf<R: Read + Seek>(
        source: GGUFSource<'_, R>,
        ctx: &LoadContext,
    ) -> Result<Self>;
}
```

### FromSafetensors

```rust
pub trait FromSafetensors: Sized + TransformerModel {
    type Config: DeserializeOwned;

    fn from_safetensors(
        source: SafetensorsSource,
        config: Self::Config,
        ctx: &LoadContext,
    ) -> Result<Self>;
}
```

---

## Loader API

### GGUFLoader

```rust
pub struct GGUFLoader;

impl GGUFLoader {
    /// Typed load when you know the model architecture
    pub fn load<M: FromGGUF, R: Read + Seek>(
        source: GGUFSource<'_, R>,
        ctx: &LoadContext,
    ) -> Result<M>;

    /// Dynamic load with architecture auto-detection
    /// Returns base model or LoRA model (both implement TransformerModel)
    pub fn load_auto<R: Read + Seek>(
        source: GGUFSource<'_, R>,
        ctx: &LoadContext,
    ) -> Result<Box<dyn TransformerModel + Send + Sync>>;

    /// Load XLoRA model (separate method, returns different trait)
    pub fn load_xlora<R: Read + Seek>(
        source: GGUFSource<'_, R>,
        ctx: &LoadContext,
        xlora_cfg: &XLoraLoadConfig,
    ) -> Result<Box<dyn XLoraModel + Send + Sync>>;
}
```

### SafetensorsLoader

```rust
pub struct SafetensorsLoader;

impl SafetensorsLoader {
    pub fn load<M: FromSafetensors>(
        source: SafetensorsSource,
        ctx: &LoadContext,
    ) -> Result<M>;

    pub fn load_auto(
        source: SafetensorsSource,
        ctx: &LoadContext,
    ) -> Result<Box<dyn TransformerModel + Send + Sync>>;
}
```

---

## Architecture Detection

### GGUF
Architecture embedded in metadata: `general.architecture`

```rust
impl GGUFSource {
    pub fn architecture(&self) -> Result<GGUFArchitecture> {
        let arch_str = self.content.get_metadata()
            .get("general.architecture")?
            .to_string()?;
        GGUFArchitecture::from_value(&arch_str)
    }
}
```

### Safetensors
Architecture in `config.json`: `architectures` array or `model_type`

```rust
impl SafetensorsSource {
    pub fn architecture(&self) -> Result<String> {
        // Try "architectures" array first (HuggingFace format)
        if let Some(arch) = self.config_json["architectures"][0].as_str() {
            return Ok(arch.to_string());
        }
        // Fall back to "model_type"
        self.config_json["model_type"].as_str()
            .map(String::from)
            .ok_or_else(|| anyhow!("Cannot detect architecture"))
    }
}
```

---

## Common Config Types

```rust
/// Common transformer config fields (Safetensors config.json)
#[derive(Deserialize)]
pub struct CommonTransformerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: Option<usize>,

    #[serde(flatten)]
    pub rope: RopeConfig,

    // Architecture detection
    pub architectures: Option<Vec<String>>,
    pub model_type: Option<String>,
}

/// Model-specific configs extend with their own fields
#[derive(Deserialize)]
pub struct LlamaConfig {
    #[serde(flatten)]
    pub common: CommonTransformerConfig,

    // Llama-specific
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
}
```

---

## Migration Path

### Phase 2: Replace Model Enum (Next)
1. Change `Model` enum in `gguf.rs` to `Box<dyn TransformerModel>`
2. Use `GGUFLoader::load_auto()` for construction
3. XLora models stay separate (use `load_xlora()`)

### Phase 3: Extract Loaders
1. Create `loaders/` module
2. Move `FromGGUF` trait to `loaders/gguf.rs`
3. Add `GGUFLoader` struct with typed and dynamic methods
4. Adapt existing model `from_gguf` implementations

### Phase 4: Safetensors Migration
1. Make Safetensors models stateless (remove cache ownership)
2. Implement `FromSafetensors` trait
3. Add `SafetensorsLoader`
4. Deprecate `NormalModel` trait

### Phase 5: Pipeline Unification
1. Create unified pipeline that works with any `TransformerModel`
2. Deprecate `GGUFPipeline` and `NormalPipeline`
3. XLora gets separate `XLoraPipeline`

---

## Open Questions (Resolved)

### 1. Loaders: Typed vs Trait Objects
**Decision**: Both. Typed `load::<M>()` when you know the type, `load_auto()` for dynamic.

### 2. Cache Ownership
**Decision**: Pipelines own cache. Models are stateless.

### 3. Model-specific Sampling
**Decision**: Metadata loaded with model, passed to pipeline via `GeneralMetadata`.

### 4. XLora Handling
**Decision**: Separate `XLoraModel` trait with two-pass inference. Separate `load_xlora()` method.

### 5. Adapter Loading
**Decision**: LoRA config in `LoadContext` (same `TransformerModel` interface). XLoRA is separate loader method returning `XLoraModel`.

---

## File Changes Summary

| Phase | Files | Changes |
|-------|-------|---------|
| 2 | `pipeline/gguf.rs` | Replace `Model` enum with trait object |
| 3 | `loaders/mod.rs` (new) | `LoadContext`, re-exports |
| 3 | `loaders/gguf.rs` (new) | `GGUFLoader`, `FromGGUF` trait |
| 3 | `loaders/config.rs` (new) | Common config types |
| 4 | `loaders/safetensors.rs` (new) | `SafetensorsLoader`, `FromSafetensors` |
| 4 | `models/*.rs` (safetensors) | Remove cache, implement `TransformerModel` |
| 5 | `pipeline/chat.rs` (new) | Unified `ChatPipeline` |
