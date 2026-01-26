# Pipeline & Loader Type Architecture

## Summary

Type-safe pipelines and loaders using Rust's type system. Typed internally, trait objects at boundaries.

**Key principle**: `TextPipeline<M: LanguageModel>` not `TextPipeline { model: Box<dyn TransformerModel> }`.

---

## Model Trait Hierarchy

```rust
/// Base trait - all models have weights on a device
pub trait Model: Send + Sync {
    fn device(&self) -> &Device;
}

/// TransformerModel - core transformer operations
/// Could be encoder, decoder, or embedding model
pub trait TransformerModel: Model {
    fn num_layers(&self) -> usize;
    fn max_seq_len(&self) -> usize;

    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>;
}

/// LanguageModel - decoder-only transformer for text generation
/// Adds lm_head for token prediction
pub trait LanguageModel: TransformerModel {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}

/// EmbeddingModel - for embeddings only (no generation)
/// Uses TransformerModel without lm_head
pub trait EmbeddingModel: TransformerModel {L
    fn pooling(&self) -> PoolingStrategy;
}

/// VisionLanguageModel - multimodal (VLM)
pub trait VisionLanguageModel: LanguageModel {
    fn encode_image(&self, image: &Tensor) -> Result<Tensor>;
}

/// DiffusionModel - image generation (completely different forward)
pub trait DiffusionModel: Model {
    fn denoise(&self, latents: Tensor, timestep: f32, ...) -> Result<Tensor>;
}
```

---

## Pipeline Types

Each pipeline type is parameterized by its model type:

```rust
/// Text generation pipeline
pub struct TextPipeline<M: LanguageModel> {
    model: M,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    cache: NormalCache,
    hook: Option<HookContainer>,
    metadata: Arc<GeneralMetadata>,
}

impl<M: LanguageModel> TextPipeline<M> {
    /// Hot path - M::transform() is monomorphized, zero vtable overhead
    pub fn forward(&mut self, input_ids: &Tensor, ctx: &TransformContext) -> Result<Tensor> {
        let hidden = self.model.embed(input_ids)?;
        let hidden = self.model.transform(hidden, ctx, &mut self.cache)?;
        self.model.lm_head(hidden)
    }
}

/// Embedding pipeline (no generation)
pub struct EmbeddingPipeline<M: EmbeddingModel> {
    model: M,
    tokenizer: Arc<Tokenizer>,
    // No cache, no chat template
}

/// Vision-language pipeline
pub struct VisionPipeline<M: VisionLanguageModel> {
    model: M,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    cache: NormalCache,
    image_processor: ImageProcessor,
}

/// Diffusion pipeline
pub struct DiffusionPipeline<M: DiffusionModel> {
    model: M,
    scheduler: DiffusionScheduler,
    vae: VAE,
}
```

---

## Pipeline Trait (For Engine)

The engine needs to hold heterogeneous pipelines. We use a trait:

```rust
/// Base trait for all pipelines - used by Engine
pub trait Pipeline: Send + Sync {
    fn forward_inputs(&mut self, inputs: Box<dyn Any>) -> Result<ForwardInputsResult>;
    fn device(&self) -> &Device;
    fn name(&self) -> &str;
}

/// Capability traits (implement only what's relevant)
pub trait Cacheable: Pipeline {
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
}

pub trait Tokenizing: Pipeline {
    fn tokenizer(&self) -> &Arc<Tokenizer>;
}

pub trait Hookable: Pipeline {
    fn set_hook(&mut self, hook: HookContainer);
    fn get_hook(&self) -> Option<&HookContainer>;
}

// Typed pipelines implement relevant capabilities
impl<M: LanguageModel> Pipeline for TextPipeline<M> { ... }
impl<M: LanguageModel> Cacheable for TextPipeline<M> { ... }
impl<M: LanguageModel> Tokenizing for TextPipeline<M> { ... }
impl<M: LanguageModel> Hookable for TextPipeline<M> { ... }
```

---

## Typed Loaders

Loaders are parameterized by model type:

```rust
/// GGUF loader - extracts metadata and loads models
pub struct GgufLoader<M> {
    paths: Vec<PathBuf>,
    metadata: GgufMetadata,
    _model: PhantomData<M>,
}

impl<M: LanguageModel + FromGGUF> GgufLoader<M> {
    /// Open GGUF files for a specific model type
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let metadata = GgufMetadata::extract(&paths)?;
        Ok(Self { paths, metadata, _model: PhantomData })
    }

    /// Load into a typed pipeline
    pub fn load(
        self,
        device: &Device,
        mapper: Box<dyn DeviceMapper>,
        attention: AttentionImplementation,
    ) -> Result<TextPipeline<M>> {
        let model = M::from_gguf(&self.paths, device, mapper, attention)?;
        let tokenizer = self.tokenizer()?;
        let chat_template = self.chat_template()?;
        let cache = NormalCache::new(model.num_layers(), model.max_seq_len());

        Ok(TextPipeline { model, tokenizer, chat_template, cache, hook: None, metadata: ... })
    }

    // Metadata accessors
    pub fn tokenizer(&self) -> Result<Arc<Tokenizer>> { ... }
    pub fn chat_template(&self) -> Result<Arc<ChatTemplate>> { ... }
    pub fn num_layers(&self) -> usize { self.metadata.num_layers }
}

/// Safetensors loader
pub struct SafetensorsLoader<M> {
    model_dir: PathBuf,
    config: serde_json::Value,
    _model: PhantomData<M>,
}

impl<M: LanguageModel + FromSafetensors> SafetensorsLoader<M> {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> { ... }
    pub fn load(self, vb: ShardedVarBuilder, ...) -> Result<TextPipeline<M>> { ... }
}
```

---

## Runtime Architecture Dispatch

When architecture is detected at runtime (user provides path, not type):

```rust
/// Dispatch based on detected architecture
/// This is the ONE place we convert typed → trait object
pub fn load_gguf_text_pipeline(
    paths: &[PathBuf],
    device: &Device,
    ...
) -> Result<Box<dyn Pipeline>> {
    let metadata = GgufMetadata::extract(paths)?;

    match metadata.architecture {
        GGUFArchitecture::Llama => {
            let pipeline = GgufLoader::<QLlama>::open(paths)?.load(device, ...)?;
            Ok(Box::new(pipeline))
        }
        GGUFArchitecture::Qwen3 => {
            let pipeline = GgufLoader::<QQwen3>::open(paths)?.load(device, ...)?;
            Ok(Box::new(pipeline))
        }
        GGUFArchitecture::Phi3 => {
            let pipeline = GgufLoader::<QPhi3>::open(paths)?.load(device, ...)?;
            Ok(Box::new(pipeline))
        }
        // ... other architectures
        arch => bail!("Unsupported architecture: {:?}", arch),
    }
}
```

---

## Benefits

1. **Type safety**: `TextPipeline<QLlama>` is a concrete type
2. **Zero-cost hot paths**: `model.transform()` is monomorphized
3. **Invalid states unrepresentable**: Can't use embedding model with TextPipeline
4. **Single dispatch point**: Only `load_gguf_text_pipeline()` uses trait objects
5. **Clear ownership**: Pipeline owns model, cache, tokenizer

---

## Migration Path

### Phase 1: Define Trait Hierarchy

1. Move `lm_head()` from `TransformerModel` to new `LanguageModel` trait
2. Update existing models to implement `LanguageModel`
3. Keep `TransformerModel` for embedding models

### Phase 2: Create Typed Pipeline

1. Create `TextPipeline<M: LanguageModel>` struct
2. Implement `Pipeline` trait for it
3. Move `forward_transformer()` logic into `TextPipeline::forward()`

### Phase 3: Create Typed Loader

1. Add type parameter to `GgufLoader<M>`
2. Implement `load() -> Result<TextPipeline<M>>`
3. Create dispatch function for runtime architecture detection

### Phase 4: Wire Up

1. Update `GGUFLoader` (the builder) to use new typed loader
2. Deprecate `GGUFPipeline` in favor of `TextPipeline<M>`
3. Update engine to use `Box<dyn Pipeline>`

### Phase 5: Other Pipeline Types

1. `EmbeddingPipeline<M: EmbeddingModel>`
2. `VisionPipeline<M: VisionLanguageModel>`
3. `DiffusionPipeline<M: DiffusionModel>`

---

## Files to Modify

| File | Changes |
|------|---------|
| `models/mod.rs` | Split `TransformerModel` → `TransformerModel` + `LanguageModel` |
| `models/quantized_*.rs` | Implement `LanguageModel` instead of just `TransformerModel` |
| `pipeline/text.rs` (new) | `TextPipeline<M>` struct |
| `pipeline/loaders/gguf_metadata.rs` | Add type parameter, `load()` method |
| `pipeline/mod.rs` | `Pipeline` trait, dispatch function |
| `pipeline/gguf.rs` | Use new typed loader, eventually deprecate |

---

## Current State (2025-01-25)

### Completed
- `GgufLoader` metadata extraction (untyped, in `pipeline/loaders/gguf_metadata.rs`)
- `TransformerModel` trait (but includes `lm_head` - needs split)
- `ModelVariant` enum (Transformer + XLora)

### Next Step
Split `TransformerModel` trait to create `LanguageModel` supertrait with `lm_head()`.
