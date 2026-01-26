# mistral.rs Type Architecture

## Philosophy

We encode domain concepts in the type system to make invalid states unrepresentable. Pure types make debugging vastly simpler—when a function's behavior depends only on its inputs, reasoning about correctness becomes tractable. We separate runtime state from configuration and stateless ML primitives that comprise the true data pipeline. This fixes encapsulation and separation of concerns that had become muddled.

The goal is an architecture flexible enough to accommodate new models and modalities without exploding line counts or over-abstracting. We thread this balance by thinking carefully about the true relationships between concepts and trusting that code quality and performance follow naturally from proper abstractions in Rust.

---

## Core Type Hierarchy

### Engine
The scheduler/runtime. Hosts multiple Pipelines, manages compute resources, dispatches Sequences.

```
Engine
  ├── Pipelines: Vec<Box<dyn Pipeline>>
  ├── Scheduler
  └── Device topology
```

### Pipeline
Runtime context for a loaded model. Owns the cache and orchestration machinery.

```
Pipeline (trait)
  ├── Model (weights + config, stateless)
  ├── Cache (KV cache, mutable runtime state)
  ├── Hooks (PP streaming, callbacks)
  └── Device placement

Variants:
  - LanguagePipeline      (decoder-only LLM)
  - DiffusionPipeline     (image generation)
  - MultimodalPipeline    (VLM/ALM with encoders)
```

### Model
Weights and configuration. Stateless—inference is a pure function of inputs.

```
Model
  ├── Config (immutable hyperparameters)
  └── Weights (tensors, immutable after load)

fn forward(&self, input, &mut cache) -> output
```

### Sequence
The data payload flowing through a Pipeline. Represents a single request/generation.

```
Sequence
  ├── Tokens / Embeddings / Images
  ├── Sampling state
  └── Request metadata
```

---

## ML Primitives

### TransformerBlock<N, A, F>
Generic pre-norm transformer layer. Zero-cost abstraction through monomorphization.

```rust
TransformerBlock<N, A, F>
where
    N: Module,        // Normalization (RmsNorm, LayerNorm, QRmsNorm)
    A: Attention,     // Attention mechanism
    F: FeedForward,   // FFN (Mlp, MoE)
```

**Pattern:**
```
input → attn_norm → attention → +residual → ffn_norm → ffn → +residual → output
```

**Usage:**
```rust
type Qwen3Block = TransformerBlock<QRmsNorm, CausalAttention, Mlp>;
type LlamaBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;
type MixtralBlock = TransformerBlock<RmsNorm, CausalAttention, MoE>;
```

### Attention (trait)
Abstracts over attention implementations.

```rust
trait Attention: Send + Sync {
    fn forward(&self, x, mask, cache, positions, metadata) -> Result<Tensor>;
    fn has_paged_attn(&self) -> bool;
}
```

**Implementations:**
- `CausalAttention` — standard causal with optional QK-norm, RoPE, paged attention
- `SlidingWindowAttention` — Mistral-style window
- `BidirectionalAttention` — encoder models

### FeedForward (trait)
Abstracts over FFN implementations.

```rust
trait FeedForward: Send + Sync {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

**Implementations:**
- `Mlp` — standard gated MLP (SiLU, GELU)
- `MoE` — mixture of experts

### CausalAttention
Configurable attention with optional features composed at construction.

```rust
CausalAttention
  ├── AttentionConfig (heads, kv_heads, head_dim)
  ├── Projections (q, k, v, o)
  ├── Optional: QkNorm
  ├── Optional: PositionEncoding (RoPE)
  ├── Optional: PagedAttention
  └── Optional: dtype conversion
```

**Builder pattern:**
```rust
CausalAttention::new(config, q, k, v, o)
    .with_qk_norm(norm)
    .with_rotary(rope)
    .with_paged_attn(paged)
```

---

## Configuration vs Runtime State

| Concept | Mutability | Lifetime | Owner |
|---------|------------|----------|-------|
| Model Config | Immutable | Static | Model |
| Weights | Immutable | Static | Model |
| KV Cache | Mutable | Per-sequence | Pipeline |
| Hooks | Mutable | Per-pipeline | Pipeline |
| Sequence State | Mutable | Per-request | Engine |

---

## Model Traits

Models define capabilities. Pipelines decide what to call.

### TransformerModel
Core transformer operations.

```rust
trait TransformerModel {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext) -> Result<Tensor>;
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
}
```

### TokenizerModel
Text encoding/decoding.

```rust
trait TokenizerModel {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>>;
    fn detokenize(&self, tokens: &[Token]) -> Result<String>;
}
```

### EmbeddingModel
For embedding-only use cases (no generation).

```rust
trait EmbeddingModel {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext) -> Result<Tensor>;
    // No lm_head — returns hidden states
}
```

---

## Pipeline Parallelism

PP is a **pipeline concern**, not a model concern. Models define capabilities; pipelines orchestrate execution.

```
HEAD stage:   embed() → transform() → send activation
MIDDLE stage: receive → transform() → send activation
TAIL stage:   receive → transform() → lm_head() → return logits
```

The pipeline knows its stage and calls the appropriate model methods. The model has no awareness of being split—it just implements pure functions over tensors.

---

## Design Principles

1. **Stateless models** — `forward()` is pure given weights and cache
2. **Composition over inheritance** — `TransformerBlock<N, A, F>` not class hierarchies
3. **Make invalid states unrepresentable** — PP stage without lm_head can't call lm_head
4. **Zero-cost abstractions** — generics monomorphize, no vtable overhead in hot paths
5. **Separation of concerns** — loading (GGUF/safetensors) separate from inference

---

## Further Reading

- **[type-driven-design.md](./type-driven-design.md)** — FP/category theory concepts in Rust. Typeclasses, sum/product types, typestate, static vs dynamic dispatch.
- **[model-ontology.md](./model-ontology.md)** — The TRUE relationships between model capabilities. When to use supertraits vs type parameters. Why current mixins are wrong.
