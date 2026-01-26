# Type-Driven Design in mistral.rs

## Philosophy

We encode domain knowledge in the type system to make invalid states unrepresentable. This document captures the theoretical foundations and practical patterns that guide our architecture.

---

## Rust's Type System: FP Concepts Realized

### Traits as Typeclasses

Rust traits are Haskell typeclasses with a key difference: they're object-safe and can be used for dynamic dispatch. This gives us two modes:

```rust
// Static dispatch (monomorphization) - like Haskell typeclasses
fn process<T: Attention>(attn: &T, x: &Tensor) -> Result<Tensor> {
    attn.forward(x, ...)  // Compiled to direct call, inlined
}

// Dynamic dispatch (trait objects) - like Java interfaces
fn process_any(attn: &dyn Attention, x: &Tensor) -> Result<Tensor> {
    attn.forward(x, ...)  // vtable lookup at runtime
}
```

**When to use which:**
- **Static**: Hot paths, inner loops, layer computation (zero-cost abstraction)
- **Dynamic**: Configuration, extensibility, user-facing APIs (flexibility)

### Sum Types (Coproducts)

Rust enums are algebraic sum types. Unlike union types in TypeScript, they're **tagged** and **exhaustive**:

```rust
pub enum ForwardInputsResult {
    CausalGeneration { logits: Tensor },
    Embeddings { embeddings: Tensor },
    Image { images: Vec<DynamicImage> },
    // ...
}

// Compiler enforces handling ALL variants
match result {
    ForwardInputsResult::CausalGeneration { logits } => ...,
    ForwardInputsResult::Embeddings { embeddings } => ...,
    // Missing arms = compile error
}
```

This is the **catamorphism** pattern: fold over a sum type by providing a handler for each case.

### Product Types (Records)

Struct fields are product types. `TransformContext` bundles related parameters:

```rust
pub struct TransformContext<'a> {
    pub seq_len: usize,
    pub position_offset: usize,
    pub paged_attn: Option<&'a PagedAttentionContext<'a>>,
}
```

This avoids growing parameter lists and groups conceptually related data.

---

## Core Patterns

### 1. Parameterized Composition (Functor-like)

`TransformerBlock<N, A, F>` is our flagship pattern. It's a **higher-kinded type** approximation in Rust:

```rust
pub struct TransformerBlock<N, A, F> {
    pub attn_norm: N,
    pub attention: A,
    pub ffn_norm: N,
    pub ffn: F,
}

impl<N, A, F> TransformerBlock<N, A, F>
where
    N: Module,
    A: Attention,
    F: FeedForward,
{
    pub fn forward(&self, hidden: Tensor, ...) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let x = self.attn_norm.forward(&hidden)?;
        let attn = self.attention.forward(&x, ...)?;
        let x = (attn + &hidden)?;

        // Pre-norm + FFN + residual
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        (x + residual)
    }
}
```

**Why this works:**
- **Zero-cost abstraction**: Monomorphization generates specialized code for each combination
- **Composition**: N, A, F are independent "slots" that can be filled with any conforming type
- **Pattern encoding**: The pre-norm transformer pattern is encoded structurally, not by convention

**Usage:**
```rust
type LlamaBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;
type MixtralBlock = TransformerBlock<RmsNorm, CausalAttention, MoeMlp>;
type Qwen3Block = TransformerBlock<QRmsNorm, CausalAttention, Mlp>;
```

### 2. Trait Hierarchy (Subtyping via Bounds)

```rust
pub trait Model: Send + Sync {
    fn device(&self) -> &Device;
}

pub trait TransformerModel: Model {
    fn num_layers(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>;
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}
```

This is **inheritance without inheritance**. `TransformerModel: Model` means every transformer model provides `device()`, but the relationship is capability-based, not identity-based.

**Key insight**: Traits define *what you can do*, not *what you are*.

### 3. Trait Composition (Current State: Anti-Pattern)

The current codebase uses "mixins" — this is a Python pattern that doesn't translate well to Rust:

```rust
// ANTI-PATTERN: Forced supertraits masquerading as capabilities
pub trait Pipeline:
    Send + Sync
    + PreProcessingMixin      // Has supertrait dep on MetadataMixin (why?)
    + IsqPipelineMixin        // GGUF returns error, barely used
    + CacheManagerMixin       // Actually useful
    + MetadataMixin           // Dumping ground: device, tokenizer, name, XLora state...
    + AnyMoePipelineMixin     // Mostly unreachable!() stubs
{
    fn forward_inputs(&mut self, ...) -> Result<ForwardInputsResult>;
}
```

**Problems:**
1. **Forced coupling** — Every pipeline must implement all mixins, even if irrelevant
2. **Violation of SRP** — `MetadataMixin` has 6 unrelated methods
3. **Stub implementations** — `AnyMoePipelineMixin` is `unreachable!()` for most pipelines
4. **Supertrait pollution** — `PreProcessingMixin: MetadataMixin` creates hidden dependencies
5. **Wrong mental model** — "Mixin" implies runtime composition; Rust traits are compile-time

**Better Rust Pattern: Optional Capability Traits**

```rust
// Minimal core trait
pub trait Pipeline: Send + Sync {
    fn forward(&mut self, inputs: ModelInputs) -> Result<ForwardOutput>;
    fn device(&self) -> &Device;
    fn name(&self) -> &str;
}

// Optional capabilities — implement only if relevant
pub trait Tokenizing: Pipeline {
    fn tokenizer(&self) -> &Tokenizer;
}

pub trait ChatCapable: Tokenizing {
    fn chat_template(&self) -> &ChatTemplate;
}

pub trait Quantizable: Pipeline {
    fn requantize(&mut self, isq_type: IsqType) -> Result<()>;
}

// Concrete pipelines implement what they need
impl Pipeline for UnifiedPipeline { ... }
impl Tokenizing for UnifiedPipeline { ... }
impl ChatCapable for UnifiedPipeline { ... }
// Does NOT implement Quantizable — and that's fine!
```

**Key difference**: Callers ask for what they need via trait bounds, not forced inheritance:

```rust
fn chat_completion<P: ChatCapable>(pipeline: &mut P, messages: Vec<Message>) -> Result<Response> {
    let template = pipeline.chat_template();  // Guaranteed by bound
    // ...
}
```

### 4. Builder Pattern (Fluent API)

```rust
CausalAttention::new(config, q_proj, k_proj, v_proj, o_proj, rope)
    .with_qk_norm(qk_norm)        // Optional: Qwen3
    .with_paged_attn(paged_attn)  // Optional: efficiency
    .with_attn_dtype(dtype)       // Optional: quantization
```

Each method returns `Self`, enabling chaining. Optional components are added incrementally.

### 5. Typestate Pattern (Compile-Time State Machines)

Not fully used yet, but the pattern:

```rust
// Phantom types encode state
struct PipelineBuilder<S: BuilderState> {
    config: Config,
    _state: PhantomData<S>,
}

// State types
struct Unconfigured;
struct ModelSelected;
struct Ready;

impl PipelineBuilder<Unconfigured> {
    fn with_model(self, model: Model) -> PipelineBuilder<ModelSelected> { ... }
}

impl PipelineBuilder<ModelSelected> {
    fn with_tokenizer(self, tok: Tokenizer) -> PipelineBuilder<Ready> { ... }
}

impl PipelineBuilder<Ready> {
    fn build(self) -> Pipeline { ... }  // Only callable when Ready
}
```

**Benefit**: The compiler prevents calling `build()` before configuration is complete.

---

## Category Theory Concepts Applied

### Functors
A functor maps objects and morphisms while preserving structure. In Rust:

```rust
impl<N, A, F> TransformerBlock<N, A, F> {
    // Maps hidden state through layers while preserving structure
    fn forward(&self, hidden: Tensor, ...) -> Result<Tensor>
}
```

The block is a functor from `Tensor` to `Tensor` that preserves the transformer computation pattern.

### Monoids
A monoid has an identity element and an associative binary operation. Residual connections are monoids:

```rust
// Identity: zero tensor
// Operation: tensor addition
let output = (attention_output + residual)?;  // Associative, identity is 0
```

### Coproducts (Sum Types)
`ForwardInputsResult` is a coproduct:

```rust
enum ForwardInputsResult {
    CausalGeneration { logits: Tensor },
    Embeddings { embeddings: Tensor },
    // ...
}
```

The coproduct provides a universal property: any function that handles each case can be expressed as a single function on the coproduct.

### Products
Structs are products. `TransformContext` is the product of `seq_len`, `position_offset`, and `paged_attn`.

### Natural Transformations
When we have a function between functors that works uniformly:

```rust
// Quantization is a natural transformation
// It transforms any QuantMethod into another while preserving the interface
fn apply_isq(self: Arc<Self>, ...) -> Result<Arc<dyn QuantMethod>>;
```

---

## When to Use What: Supertraits vs Type Parameters vs Optional Traits

This is the key design decision. Get it wrong and you end up with forced inheritance or missing capabilities.

### Use Supertraits When: One Capability REQUIRES Another

The relationship is **essential**, not incidental. You literally cannot implement the subtrait without the supertrait.

```rust
// ✅ CORRECT: Autoregressive generation requires a vocabulary to sample from
trait Autoregressive: Tokenizing {
    fn sample(&self, logits: &Tensor, params: &SamplingParams) -> Result<u32>;
}

// ✅ CORRECT: TransformerModel IS-A Model (all transformers have device)
trait TransformerModel: Model {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    // ...
}

// ✅ CORRECT: LanguageModel combines all capabilities needed for text generation
trait LanguageModel: TransformerModel + Tokenizing + Autoregressive + Cacheable {}
```

**Note:** Chat templates are CONFIGURATION, not a capability trait. All language models complete prompts — "chat" is just message formatting. See model-ontology.md.

### DON'T Use Supertraits When: Relationship is Incidental

```rust
// ❌ WRONG: Not all tokenizing models use cache
trait Tokenizing: Cacheable { ... }  // Embedding models tokenize but don't cache!

// ❌ WRONG: MetadataMixin shouldn't be required for tokenization
trait PreProcessingMixin: MetadataMixin { ... }  // Why?

// ❌ WRONG: Pipeline shouldn't force all 5 mixins
trait Pipeline: A + B + C + D + E { ... }  // Embedding pipelines don't need CacheManagerMixin!
```

### Use Type Parameters When: Implementation Varies Independently

The components can be mixed and matched. Each "slot" is filled independently.

```rust
// ✅ CORRECT: N, A, F are independent implementation choices
struct TransformerBlock<N, A, F> {
    attn_norm: N,    // Could be RmsNorm, LayerNorm, QRmsNorm
    attention: A,    // Could be Causal, SlidingWindow, Bidirectional
    ffn: F,          // Could be Mlp, MoeMlp, GatedMlp
}

// ✅ CORRECT: Cache strategy varies independently of model
struct Pipeline<M: Model, C: CacheStrategy> {
    model: M,
    cache: C,
}

// ✅ CORRECT: Quantization method varies independently
struct Linear<Q: QuantMethod> {
    weights: Q,
}
```

### Use Optional Traits When: Capability is Truly Optional

Callers specify what they need via trait bounds. Types implement what they support.

```rust
// ✅ CORRECT: Caller asks for what it needs
fn generate<P: LanguageModel>(pipeline: &mut P) -> Result<String> {
    // Guaranteed: P has all language model capabilities
}

fn embed<P: Tokenizing>(pipeline: &P) -> Result<Tensor> {
    // Only needs tokenizer, no cache required
}

fn process_vision<P: LanguageModel + VisionEncoding>(pipeline: &mut P, image: &Image) -> Result<String> {
    // Needs both language AND vision capabilities
}

// Types implement what they actually support
impl LanguageModel for TextPipeline { ... }  // Full language model
impl Tokenizing for EmbeddingPipeline { ... }  // Only tokenizes, no generation

// Chat is handled via configuration, not traits:
impl TextPipeline {
    fn chat(&mut self, messages: &[Message]) -> Result<String> {
        let prompt = self.chat_template.apply(messages)?;  // Config, not trait
        self.generate(&prompt)
    }
}
```

### Real Examples from Model Ontology

| Relationship | Pattern | Reasoning |
|--------------|---------|-----------|
| `Autoregressive: Tokenizing` | Supertrait | Can't sample tokens without vocabulary |
| `LanguageModel: TransformerModel + Tokenizing + Autoregressive + Cacheable` | Supertrait | All four are essential for text generation |
| `TransformerModel: Model` | Supertrait | All transformers have weights on device |
| `TransformerBlock<N, A, F>` | Type params | Norm, attention, FFN vary independently |
| `VisionEncoding` | Optional trait | Only VLMs encode images |
| `Requantizable` | Optional trait | Not all models support runtime requant |
| Chat template | Configuration | All LMs complete prompts; chat is just formatting |

See **[model-ontology.md](./model-ontology.md)** for the full capability matrix.

---

## Design Principles

### 1. Make Invalid States Unrepresentable

**Bad:**
```rust
struct Model {
    layers: Vec<Layer>,
    is_initialized: bool,  // Runtime flag
}
```

**Good:**
```rust
struct UninitializedModel { config: Config }
struct Model { layers: Vec<Layer> }  // Only exists when initialized

impl UninitializedModel {
    fn initialize(self, device: &Device) -> Model { ... }
}
```

### 2. Prefer Static Over Dynamic Dispatch in Hot Paths

```rust
// Hot path (layer computation): monomorphized
impl<N, A, F> TransformerBlock<N, A, F> where N: Module, A: Attention, F: FeedForward {
    fn forward(&self, ...) -> Result<Tensor> { ... }  // Inlined, specialized
}

// Cold path (configuration): trait objects
struct Pipeline {
    model: Box<dyn TransformerModel>,  // Runtime dispatch is fine here
}
```

### 3. Separate Data from Behavior

**Models are stateless**:
```rust
// Model holds weights (immutable)
impl TransformerModel for LlamaModel {
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>
    // Cache passed in, not owned
}

// Pipeline holds state (mutable)
struct Pipeline {
    model: Box<dyn TransformerModel>,
    cache: EitherCache,  // Owned by pipeline
}
```

### 4. Encode Invariants in Types

```rust
// Instead of runtime checks:
fn forward(&self, hidden: Tensor) -> Result<Tensor> {
    assert!(hidden.dims().len() == 3);  // Runtime panic
    ...
}

// Encode in types (when practical):
struct HiddenStates<const B: usize, const S: usize, const H: usize>(Tensor);
// Shape enforced at compile time (where B=batch, S=seq, H=hidden)
```

### 5. Composition Over Inheritance

```rust
// Bad: deep inheritance
class MixtralModel extends TransformerModel extends LanguageModel extends Model { ... }

// Good: composition via generics
struct MixtralModel {
    blocks: Vec<TransformerBlock<RmsNorm, CausalAttention, MoeMlp>>,
}
```

---

## Opportunities for Further Improvement

### 1. Full Typestate for Builders
Current builders use simple method chaining. Full typestate would make incomplete configuration a compile error.

### 2. Associated Types for Cache Format
Currently `KvCache` is an enum. Associated types could enable cache-format-specific optimizations:

```rust
trait AttentionBackend {
    type Cache: KvCache;
    type Mask: AttentionMask;
}
```

### 3. GADT-like Patterns for Model Dispatch
Using trait objects loses type information. We could preserve more:

```rust
enum Model<Arch> {
    Llama(LlamaModel<Arch>),
    Mistral(MistralModel<Arch>),
}
```

### 4. Sealed Traits for Safety
Prevent downstream crates from implementing internal traits:

```rust
mod private { pub trait Sealed {} }
pub trait QuantMethod: private::Sealed { ... }
```

---

## Summary

| Concept | Rust Feature | Usage in mistral.rs |
|---------|--------------|---------------------|
| Typeclasses | Traits | `Attention`, `FeedForward`, `TransformerModel` |
| Sum Types | Enums | `ForwardInputsResult`, `AttentionPattern` |
| Product Types | Structs | `TransformContext`, `AttentionConfig` |
| Higher-Kinded Types | Generic params | `TransformerBlock<N, A, F>` |
| Subtyping | Trait bounds | `TransformerModel: Model` |
| Horizontal Composition | Multiple bounds | `Pipeline: PreProcessingMixin + ...` |
| Typestate | PhantomData | (Opportunity for builders) |
| Monomorphization | Generics | Layer computation paths |
| Dynamic Dispatch | `dyn Trait` | Plugin points, configuration |

The key insight: **Rust's type system is expressive enough to encode domain invariants, architectural patterns, and capability requirements at compile time.** We leverage this to catch errors early, enable zero-cost abstractions, and create self-documenting code.
