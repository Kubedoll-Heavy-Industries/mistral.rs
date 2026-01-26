# Model Ontology: The True Relationships

## Core Insight

The relationships between model capabilities are **not what the current mixin structure implies**. Here's what's actually true:

---

## Capability Matrix

| Model Type | Tokenizes | Autoregressive | KV Cache | Chat Template | XLoRA Support | MoE |
|------------|-----------|----------------|----------|---------------|---------------|-----|
| **Language (Llama, Mistral, Qwen)** | Yes | Yes | Yes | Yes | Yes | No |
| **Language MoE (Mixtral, Qwen3MoE)** | Yes | Yes | Yes | Yes | Yes | Built-in |
| **Embedding (BERT, Qwen-Embed)** | Yes | **No** | **No** | **No** | No | No |
| **Vision-Language (LLaVA, Qwen-VL)** | Yes | Yes | Yes | Yes | Yes | Some |
| **Diffusion (Flux, SD)** | **No** | **No** | **No** | **No** | No | No |
| **Speech** | **No** | **No** | **No** | **No** | No | No |

---

## The Real Hierarchy

```
Model (weights on device)
│
├── TransformerModel (embed → transform → lm_head)
│   │
│   ├── LanguageModel : TransformerModel + Tokenizing + Autoregressive + Cacheable
│   │   ├── Standard (Llama, Mistral, Qwen, Phi, Gemma, ...)
│   │   └── MoE variants (Mixtral, Qwen3MoE) — MoE is an FFN implementation detail
│   │
│   ├── EmbeddingModel : TransformerModel + Tokenizing
│   │   └── NO lm_head call, NO cache, NO generation
│   │   └── Returns hidden states, not logits
│   │
│   └── VisionLanguageModel : TransformerModel + Tokenizing + Autoregressive + Cacheable + VisionEncoding
│       └── Additional image encoder, multimodal fusion
│
├── DiffusionModel (denoise latents)
│   └── NO tokenizer, NO autoregressive, iterative refinement
│
└── SpeechModel (audio processing)
    └── NO tokenizer (in the text sense), different modality
```

---

## Capability Traits (Orthogonal Concerns)

### Core Model Traits

```rust
/// All models have weights on a device
trait Model: Send + Sync {
    fn device(&self) -> &Device;
}

/// Transformer architecture (most LLMs)
trait TransformerModel: Model {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor>;
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor>;
    fn num_layers(&self) -> usize;
    fn max_seq_len(&self) -> usize;
}
```

### Capability Traits (Compose as Needed)

```rust
/// Can encode/decode text
trait Tokenizing {
    fn tokenizer(&self) -> &Tokenizer;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

/// Generates output autoregressively (one token at a time)
trait Autoregressive: Tokenizing {
    fn sample(&self, logits: &Tensor, params: &SamplingParams) -> Result<u32>;
}

/// Uses KV cache for efficient sequential generation
trait Cacheable {
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
}

/// Can process images alongside text
trait VisionEncoding {
    fn encode_image(&self, image: &DynamicImage) -> Result<Tensor>;
}

/// Supports runtime requantization
trait Requantizable {
    fn requantize(&mut self, dtype: IsqType) -> Result<()>;
}
```

### Configuration, NOT Traits

```rust
// Chat templates are CONFIGURATION, not capability
// All language models can complete prompts — "chat" is just formatting
struct LanguagePipeline {
    model: Box<dyn LanguageModel>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Option<ChatTemplate>,  // Config, not trait
    cache: EitherCache,
}

// The model doesn't know it's "chatting" — it just completes tokens
impl LanguagePipeline {
    fn generate(&mut self, prompt: &str) -> Result<String> { ... }

    fn chat(&mut self, messages: &[Message]) -> Result<String> {
        let prompt = self.chat_template
            .as_ref()
            .ok_or(anyhow!("No chat template configured"))?
            .apply(messages)?;
        self.generate(&prompt)
    }
}
```

**Why not a `ChatCapable` trait?**
- All language models complete prompts — that's the definition
- "Chat" is preprocessing (format messages → prompt string)
- Whether a model follows chat format well depends on training, not runtime capability
- Chat template is tokenizer/config concern, not model capability

---

## XLoRA: A Modifier, Not a Model Type

XLoRA is **not a separate model architecture**. It's a runtime modifier:

```rust
/// XLoRA-enabled models have a different forward signature
trait XLoraCapable: TransformerModel {
    /// Forward with adapter routing
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,  // Full sequence for adapter selection
        state: &mut NonGranularState,
        // ... other params
    ) -> Result<Tensor>;
}
```

**Key facts:**
- Same base model weights
- Additional LoRA adapter weights (external files)
- Classifier network selects which adapter per token
- Requires `input_ids_full` for adapter selection (different signature!)
- Stored as flag: `metadata.is_xlora: bool`

**Implication:** XLoRA models need either:
1. A separate trait with different forward signature, OR
2. A type parameter `Pipeline<M, Adapter = NoAdapter>` with typestate

---

## MoE: An FFN Implementation Detail

MoE (Mixture of Experts) is **not a pipeline concern**. It's an internal architecture detail:

```rust
// MoE is just a different FeedForward implementation
impl FeedForward for MoeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gates = self.router.forward(xs)?;  // Select experts
        let expert_outputs = self.experts.forward(xs, &gates)?;
        self.aggregate(expert_outputs, &gates)
    }
}

// Used in TransformerBlock like any other FFN
type MixtralBlock = TransformerBlock<RmsNorm, CausalAttention, MoeMlp>;
type LlamaBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;
// ↑ Same block structure, different FFN
```

**Two kinds of MoE:**
1. **Built-in** (Mixtral, Qwen3MoE) — Architecture defines expert routing
2. **Trained (AnyMoE)** — Train router to convert dense FFN → expert selection

**Implication:** MoE shouldn't be a mixin on Pipeline. It's already handled by `TransformerBlock<N, A, F>` where F can be `Mlp` or `MoeMlp`.

---

## What the Current Mixins Actually Are

| Mixin | Reality | Should Be |
|-------|---------|-----------|
| `PreProcessingMixin` | Tokenizer + chat template | Split: `Tokenizing` + `ChatCapable` |
| `MetadataMixin` | Dumping ground (6 unrelated methods) | Split into focused traits |
| `CacheManagerMixin` | Batch KV cache orchestration | `Cacheable` trait, only for autoregressive |
| `IsqPipelineMixin` | Runtime requantization | `Requantizable`, optional |
| `AnyMoePipelineMixin` | Expert training hooks | Separate concern, not a pipeline mixin |

---

## Proposed Supertrait Relationships

These relationships are **essential**, not incidental:

```
Autoregressive : Tokenizing
    ↳ You can't sample tokens if you don't have a vocabulary

LanguageModel : TransformerModel + Tokenizing + Autoregressive + Cacheable
    ↳ All four are essential for text generation

VisionLanguageModel : LanguageModel + VisionEncoding
    ↳ VLMs are language models that also encode images
```

These are **NOT** supertrait relationships:

```
Pipeline : MetadataMixin + CacheManagerMixin + ...  ❌
    ↳ Not all pipelines need all capabilities

Tokenizing : Cacheable  ❌
    ↳ Embedding models tokenize but don't cache

ChatCapable : Tokenizing  ❌
    ↳ "Chat" is formatting, not capability. All LMs complete prompts.
```

**Configuration vs Capability:**
```
Chat template     → Configuration (stored in pipeline, not a trait)
Tokenizer         → Capability (Tokenizing trait)
Sampling params   → Configuration (passed to generate())
Cache strategy    → Configuration (EitherCache type)
```

---

## Type Parameters vs Supertraits

**Use supertraits when:** One capability *requires* another to function
```rust
trait Autoregressive: Tokenizing { ... }  // Can't sample without vocab
trait ChatCapable: Tokenizing { ... }      // Can't apply template without tokenizer
```

**Use type parameters when:** Implementation varies independently
```rust
struct TransformerBlock<N, A, F> { ... }  // N, A, F vary independently
struct Pipeline<M: Model, C: CacheStrategy> { ... }  // Model and cache vary independently
```

**Use optional traits when:** Capability is truly optional
```rust
// Caller specifies what they need
fn generate<P: Autoregressive + Cacheable>(pipeline: &mut P) { ... }
fn embed<P: Tokenizing>(pipeline: &P) { ... }  // No cache needed
```

---

## Summary

1. **Tokenizing** is NOT universal — diffusion/speech don't tokenize
2. **Autoregressive** implies **Tokenizing** (supertrait relationship)
3. **Cacheable** is only for autoregressive models (embedding models skip cache)
4. **MoE** is an FFN variant, not a pipeline concern — already handled by `TransformerBlock<N, A, F>`
5. **XLoRA** has a different forward signature — needs separate trait or type parameter
6. **Current mixins violate SRP** — MetadataMixin has 6 unrelated methods
7. **Forced supertraits are wrong** — not all pipelines need all capabilities
