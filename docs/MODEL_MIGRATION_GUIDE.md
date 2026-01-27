# Model Migration Guide: StandardTransformerBlock Infrastructure

This document outlines the modifications needed to migrate quantized model implementations
to use the generic `StandardTransformerBlock` infrastructure.

## Migration Status Summary

| Model | Status | Block Type | Notes |
|-------|--------|-----------|-------|
| `quantized_qwen3.rs` | ✅ COMPLETE | `StandardTransformerBlock` | Reference implementation |
| `quantized_qwen.rs` | ✅ COMPLETE | `StandardTransformerBlock` | Qwen2, optional Q/K norms |
| `quantized_mistral3.rs` | ✅ COMPLETE | `StandardTransformerBlock` | YaRN RoPE scaling |
| `quantized_starcoder2.rs` | ✅ COMPLETE | `TransformerBlock<LayerNorm, CausalAttention, NonGatedMlp>` | LayerNorm + GELU |
| `quantized_phi3.rs` | ✅ COMPLETE | `TransformerBlock<RmsNorm, FusedQkvCausalAttention, FusedGatedMlp>` | Sliding window |
| `quantized_phi2.rs` | ✅ COMPLETE | `ParallelTransformerBlock<LayerNorm, FusedQkvCausalAttention, NonGatedMlp>` | Parallel arch + partial RoPE |
| `quantized_llama.rs` | ⏸️ DEFERRED | N/A | Deprecated, xlora dependency |
| `quantized_qwen3_moe.rs` | ⏸️ DEFERRED | N/A | MoE infrastructure needed |

---

## Target Design Patterns

The goal is to consolidate all model implementations to use a common infrastructure that:
1. Reduces code duplication across model families
2. Provides unified GGUF and safetensors loading
3. Enables consistent trait implementations for typed pipelines
4. Simplifies maintenance and feature additions

### Reference Implementation: `quantized_qwen3.rs`

The `quantized_qwen3.rs` file serves as the gold standard. Key patterns:

```rust
// 1. Use StandardTransformerBlock (or a typed alias for variants)
type StandardTransformerBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;

// 2. Minimal struct - just the essential fields
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<StandardTransformerBlock>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    device: Device,
    max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

// 3. Implement FromGGUF with load_transformer_layers()
impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R>(...) -> Result<Self> {
        let config = TransformerConfig::from_gguf_metadata(&metadata)?;
        let layers = load_transformer_layers(
            &config, &mut weights, &naming, layer_range,
            &*mapper, device, attention_mechanism, dtype,
            |ctx, builder, weights| {
                // Model-specific customization (Q/K norm, etc.)
                Ok(builder)
            },
        )?;
        // ...
    }
}

// 4. Implement FromSafetensors with load_transformer_from_safetensors()
impl ModelConfig::FromSafetensors for ModelWeights {
    fn from_safetensors(...) -> Result<Self> {
        let loaded = load_transformer_from_safetensors(
            cfg, config, vb, device, &*mapper,
            attention_mechanism, dtype, layer_range,
            |ctx, builder, weights| { /* customization */ },
        )?;
        // ...
    }
}

// 5. Use standard trait helpers
impl TransformerModel for ModelWeights {
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        standard_embed(self, tokens)
    }
    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
        standard_transform(self, hidden, ctx, cache)
    }
}

impl LanguageModel for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        standard_lm_head(self, hidden)
    }
}
```

### Required Trait Implementations

Every migrated model must implement:

| Trait | Purpose | Key Methods |
|-------|---------|-------------|
| `Model` | Base trait | `device()` |
| `TransformerModel` | Core transformer ops | `num_layers()`, `max_seq_len()`, `embed()`, `transform()` |
| `TransformerModelExt` | Typed accessors | `tok_embeddings()`, `layers()`, `output_norm()`, `mapper()`, `model_dtype()` |
| `LanguageModel` | LM head | `lm_head()` |
| `LanguageModelExt` | Output accessor | `output()` |
| `LanguageModelConfig` | Config trait (for safetensors Config struct) | `hidden_size()`, `num_layers()`, etc. |

---

## Block Type Variants

The infrastructure supports multiple block variants for different architectures:

### Standard Pre-Norm Block (Most Models)

```rust
/// Standard pre-norm transformer: attn_norm → attention → residual → ffn_norm → ffn → residual
type StandardTransformerBlock = TransformerBlock<RmsNorm, CausalAttention, Mlp>;
```

Used by: Llama, Qwen, Mistral, etc.

### LayerNorm Block (StarCoder2)

```rust
/// Like standard but with LayerNorm instead of RmsNorm
type StarCoder2Block = TransformerBlock<LayerNorm, CausalAttention, NonGatedMlp>;
```

Used by: StarCoder2, GPT-2 family

### Fused Attention Block (Phi3)

```rust
/// Fused QKV projection with fused gate+up MLP
type Phi3Block = TransformerBlock<RmsNorm, FusedQkvCausalAttention, FusedGatedMlp>;
```

Used by: Phi3

### Parallel Block (Phi2)

```rust
/// Parallel architecture: attention and MLP computed in parallel from same normalized input
/// x_out = x + attention(norm(x)) + mlp(norm(x))
type Phi2Block = ParallelTransformerBlock<LayerNorm, FusedQkvCausalAttention, NonGatedMlp>;
```

Used by: Phi2

The `ParallelTransformerBlock` uses a single normalization layer and computes attention and MLP
in parallel, then sums their outputs. This differs from the standard sequential pre-norm pattern.

---

## Special Infrastructure Components

### PartialRotaryEmbedding (Phi2)

Phi2 applies RoPE only to the first `rope_dim` dimensions of the head:

```rust
/// Partial rotary embedding that applies RoPE to only part of the head dimension.
pub struct PartialRotaryEmbedding {
    inner: RotaryEmbedding,
    rope_dim: usize,
}

impl PartialRotaryEmbedding {
    pub fn new(base: f32, rope_dim: usize, max_seq_len: usize, device: &Device, is_gpt_neox: bool, dtype: DType) -> Result<Self>;

    pub fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offsets: &[usize]) -> Result<(Tensor, Tensor)> {
        // Split into rotated and pass-through parts
        let q_rot = q.narrow(D::Minus1, 0, self.rope_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rope_dim, head_dim - self.rope_dim)?;
        // Apply RoPE to rotated parts
        let (q_rotated, k_rotated) = self.inner.forward(&q_rot, &k_rot, seqlen_offsets)?;
        // Concatenate back
        Tensor::cat(&[&q_rotated, &q_pass], D::Minus1)
    }
}
```

### Sliding Window Attention (Phi3)

Phi3 uses sliding window attention which requires a different mask:

```rust
// Phi3 cannot use standard_transform() because it needs sliding window mask
fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
    // Create sliding window causal mask (different from standard)
    let mask = CausalMasker.make_sliding_window_causal_mask_matrix(
        &mask_shape,
        past_kv_len_cache,
        Some(self.max_seq_len),  // window size
        self.dtype,
        self.n_heads,
    )?;
    // ... custom layer loop
}
```

---

## Models Requiring Migration (DEFERRED)

### `quantized_llama.rs` - DEFERRED (Deprecated)

**Status:** Already marked `#[deprecated]`. Has xlora dependency that prevents deletion.

**Future Plan:**
1. Migrate xlora models to use the new infrastructure
2. Then delete `quantized_llama.rs`
3. `llama.rs` is now the universal implementation

### `quantized_qwen3_moe.rs` - DEFERRED (MoE)

**Status:** MoE (Mixture of Experts) model requiring specialized infrastructure.

**Future Plan:**
1. Create `MoeTransformerBlock<Norm, Attention, Expert>` or similar
2. Create `MoeMlp` component with expert routing
3. Then migrate using pattern:
   ```rust
   type Qwen3MoeBlock = TransformerBlock<RmsNorm, CausalAttention, MoeMlp<Mlp>>;
   ```

---

## Infrastructure Status

| Feature | Status | Used By |
|---------|--------|---------|
| `TransformerBlock<N, A, F>` | ✅ Complete | Most models |
| `ParallelTransformerBlock<N, A, F>` | ✅ Complete | Phi2 |
| `CausalAttention` | ✅ Complete | Most models |
| `FusedQkvCausalAttention` | ✅ Complete | Phi2, Phi3 |
| `Mlp` (gated SiLU) | ✅ Complete | Most models |
| `NonGatedMlp` (GELU) | ✅ Complete | StarCoder2, Phi2 |
| `FusedGatedMlp` | ✅ Complete | Phi3 |
| `RmsNorm` | ✅ Complete | Most models |
| `LayerNorm` | ✅ Complete | StarCoder2, Phi2 |
| `RotaryEmbedding` | ✅ Complete | Most models |
| `PartialRotaryEmbedding` | ✅ Complete | Phi2 |
| `Mistral3RotaryEmbedding` (YaRN) | ✅ Complete | Mistral3 |
| `standard_transform()` | ✅ Complete | Most models |
| Sliding window in `standard_transform()` | ❌ Not needed | Phi3 uses custom |
| `load_transformer_layers()` | ✅ Complete | GGUF loading |
| `load_transformer_from_safetensors()` | ✅ Complete | Safetensors loading |
| MoE support | ❌ Pending | Qwen3MoE, Mixtral |

---

## Validation Checklist

For each migrated model, verify:

- [ ] `cargo check -p mistralrs-core` passes
- [ ] `cargo clippy -p mistralrs-core -- -D warnings` passes
- [ ] `cargo test -p mistralrs-core --lib` passes
- [ ] Model loads from GGUF format
- [ ] Model loads from safetensors format (if implemented)
- [ ] Inference produces correct output (compare to reference)
- [ ] Pipeline parallelism works (layer range loading)
- [ ] Paged attention works (if supported)

---

## Adding New Model Support

To add support for a new model architecture:

1. **Identify the block type** needed based on architecture:
   - Standard pre-norm → `TransformerBlock<RmsNorm, CausalAttention, Mlp>`
   - Parallel attention+MLP → `ParallelTransformerBlock<N, A, F>`
   - Custom components → Create appropriate type alias

2. **Check for special requirements:**
   - Partial RoPE → Use `PartialRotaryEmbedding`
   - Sliding window → Implement custom `transform()`
   - Fused projections → Use `FusedQkvCausalAttention`, `FusedGatedMlp`
   - LayerNorm → Use `TransformerBlock<LayerNorm, ...>`

3. **Implement the model:**
   - Copy `quantized_qwen3.rs` as template
   - Modify block type and loading logic
   - Add customizer closure for model-specific weights

4. **Add tests and validate** per checklist above
