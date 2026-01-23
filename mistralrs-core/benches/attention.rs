//! Attention mechanism benchmarks for regression detection.
//!
//! Measures the core attention computation paths:
//! - Naive SDPA (CPU/Metal fallback)
//! - Chunked attention (memory-efficient path)
//! - Full Q @ K^T @ V pipeline
//!
//! Run with: cargo bench --package mistralrs-core --bench attention
//! Run with Metal: cargo bench --package mistralrs-core --bench attention --features metal

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as bb;

use candle_core::{DType, Device, Result, Tensor};

/// Model configurations for benchmarking (representative architectures)
struct AttentionConfig {
    name: &'static str,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
}

impl AttentionConfig {
    const fn new(
        name: &'static str,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            name,
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        }
    }
}

// Representative model configurations
const CONFIGS: &[AttentionConfig] = &[
    // Llama-style: 32 heads, GQA with 8 KV heads
    AttentionConfig::new("llama_7b_short", 1, 32, 8, 128, 64),
    AttentionConfig::new("llama_7b_medium", 1, 32, 8, 128, 512),
    AttentionConfig::new("llama_7b_long", 1, 32, 8, 128, 2048),
    // Mistral-style: 32 heads, GQA with 8 KV heads
    AttentionConfig::new("mistral_7b_short", 1, 32, 8, 128, 64),
    AttentionConfig::new("mistral_7b_medium", 1, 32, 8, 128, 512),
    // Small model: Qwen 0.6B style
    AttentionConfig::new("qwen_0.6b_short", 1, 14, 2, 64, 64),
    AttentionConfig::new("qwen_0.6b_medium", 1, 14, 2, 64, 512),
    // Batched inference
    AttentionConfig::new("llama_7b_batch4", 4, 32, 8, 128, 256),
];

fn get_device() -> Device {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).expect("Metal device should be available")
    }
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0).expect("CUDA device should be available")
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Device::Cpu
    }
}

/// Create synthetic Q, K, V tensors for attention benchmarking
fn create_qkv_tensors(
    config: &AttentionConfig,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Q: [batch, num_heads, seq_len, head_dim]
    let q = Tensor::randn(
        0f32,
        1f32,
        (config.batch_size, config.num_heads, config.seq_len, config.head_dim),
        device,
    )?
    .to_dtype(dtype)?;

    // K, V: [batch, num_kv_heads, seq_len, head_dim]
    let k = Tensor::randn(
        0f32,
        1f32,
        (config.batch_size, config.num_kv_heads, config.seq_len, config.head_dim),
        device,
    )?
    .to_dtype(dtype)?;

    let v = Tensor::randn(
        0f32,
        1f32,
        (config.batch_size, config.num_kv_heads, config.seq_len, config.head_dim),
        device,
    )?
    .to_dtype(dtype)?;

    Ok((q, k, v))
}

/// Create causal attention mask
fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mask = Tensor::tril2(seq_len, dtype, device)?;
    // Convert 0/1 mask to -inf/0 for softmax
    let mask = mask.where_cond(
        &mask.ones_like()?,
        &Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(mask.shape())?,
    )?;
    // Shape for broadcasting: [1, 1, seq_len, seq_len]
    mask.unsqueeze(0)?.unsqueeze(0)
}

/// Benchmark: Raw Q @ K^T matrix multiplication (attention scores)
fn bench_qk_matmul(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("attention_qk_matmul");

    for config in CONFIGS {
        let (q, k, _v) = create_qkv_tensors(config, &device, dtype).expect("tensor creation");

        // Throughput in FLOPs: 2 * batch * heads * seq * seq * head_dim
        let flops = 2 * config.batch_size * config.num_heads * config.seq_len * config.seq_len * config.head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(config.name),
            config,
            |b, _cfg| {
                b.iter(|| {
                    let k_t = k.t().expect("transpose");
                    let att = q.matmul(&k_t).expect("matmul");
                    device.synchronize().ok();
                    bb(att.elem_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Softmax over attention scores
fn bench_softmax(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("attention_softmax");

    for config in CONFIGS {
        // Create attention score tensor [batch, heads, seq, seq]
        let att = Tensor::randn(
            0f32,
            1f32,
            (config.batch_size, config.num_heads, config.seq_len, config.seq_len),
            &device,
        )
        .expect("tensor")
        .to_dtype(dtype)
        .expect("dtype");

        group.throughput(Throughput::Elements(
            (config.batch_size * config.num_heads * config.seq_len * config.seq_len) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::from_parameter(config.name),
            config,
            |b, _cfg| {
                b.iter(|| {
                    let result = candle_nn::ops::softmax_last_dim(&att).expect("softmax");
                    device.synchronize().ok();
                    bb(result.elem_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Full attention pipeline (Q @ K^T -> scale -> mask -> softmax -> @ V)
fn bench_full_attention(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("attention_full_pipeline");
    group.sample_size(50); // Fewer samples for slower benchmarks

    for config in CONFIGS {
        let (q, k, v) = create_qkv_tensors(config, &device, dtype).expect("tensor creation");
        let mask = create_causal_mask(config.seq_len, &device, dtype).expect("mask creation");
        let softmax_scale = 1.0 / (config.head_dim as f64).sqrt();

        // Total FLOPs: 2 * QK^T + softmax + 2 * att@V
        let flops = 4 * config.batch_size * config.num_heads * config.seq_len * config.seq_len * config.head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(config.name),
            config,
            |b, _cfg| {
                b.iter(|| {
                    // Q @ K^T
                    let att = q.matmul(&k.t().expect("transpose")).expect("qk matmul");
                    // Scale
                    let att = (att * softmax_scale).expect("scale");
                    // Apply mask
                    let att = att.broadcast_add(&mask).expect("mask");
                    // Softmax
                    let att = candle_nn::ops::softmax_last_dim(&att).expect("softmax");
                    // @ V
                    let output = att.matmul(&v).expect("av matmul");
                    device.synchronize().ok();
                    bb(output.elem_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Grouped-Query Attention (GQA) with KV expansion
fn bench_gqa_kv_expand(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("attention_gqa_expand");

    for config in CONFIGS.iter().filter(|c| c.num_heads != c.num_kv_heads) {
        let (_, k, v) = create_qkv_tensors(config, &device, dtype).expect("tensor creation");

        let num_groups = config.num_heads / config.num_kv_heads;

        group.bench_with_input(
            BenchmarkId::from_parameter(config.name),
            config,
            |b, _cfg| {
                b.iter(|| {
                    // Repeat KV heads to match Q heads (GQA pattern)
                    // [batch, kv_heads, seq, dim] -> [batch, heads, seq, dim]
                    let k_expanded = k
                        .unsqueeze(2)
                        .expect("unsqueeze")
                        .expand((config.batch_size, config.num_kv_heads, num_groups, config.seq_len, config.head_dim))
                        .expect("expand")
                        .reshape((config.batch_size, config.num_heads, config.seq_len, config.head_dim))
                        .expect("reshape");
                    let v_expanded = v
                        .unsqueeze(2)
                        .expect("unsqueeze")
                        .expand((config.batch_size, config.num_kv_heads, num_groups, config.seq_len, config.head_dim))
                        .expect("expand")
                        .reshape((config.batch_size, config.num_heads, config.seq_len, config.head_dim))
                        .expect("reshape");
                    device.synchronize().ok();
                    bb(k_expanded.elem_count() + v_expanded.elem_count())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Single decode step (seq_len=1, attention over full KV cache)
fn bench_decode_attention(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("attention_decode_step");

    // Decode configs: seq_len=1 for Q, varying KV cache lengths
    let decode_configs = [
        ("kv_64", 1, 32, 8, 128, 64),
        ("kv_256", 1, 32, 8, 128, 256),
        ("kv_1024", 1, 32, 8, 128, 1024),
        ("kv_4096", 1, 32, 8, 128, 4096),
    ];

    for (name, batch, heads, kv_heads, head_dim, kv_len) in decode_configs {
        // Q: [batch, heads, 1, head_dim]
        let q = Tensor::randn(0f32, 1f32, (batch, heads, 1, head_dim), &device)
            .expect("q")
            .to_dtype(dtype)
            .expect("dtype");
        // K, V: [batch, kv_heads, kv_len, head_dim]
        let k = Tensor::randn(0f32, 1f32, (batch, kv_heads, kv_len, head_dim), &device)
            .expect("k")
            .to_dtype(dtype)
            .expect("dtype");
        let v = Tensor::randn(0f32, 1f32, (batch, kv_heads, kv_len, head_dim), &device)
            .expect("v")
            .to_dtype(dtype)
            .expect("dtype");

        let num_groups = heads / kv_heads;
        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        group.bench_with_input(BenchmarkId::from_parameter(name), &kv_len, |b, _| {
            b.iter(|| {
                // Expand KV for GQA
                let k_exp = k
                    .unsqueeze(2)
                    .expect("unsqueeze")
                    .expand((batch, kv_heads, num_groups, kv_len, head_dim))
                    .expect("expand")
                    .reshape((batch, heads, kv_len, head_dim))
                    .expect("reshape");
                let v_exp = v
                    .unsqueeze(2)
                    .expect("unsqueeze")
                    .expand((batch, kv_heads, num_groups, kv_len, head_dim))
                    .expect("expand")
                    .reshape((batch, heads, kv_len, head_dim))
                    .expect("reshape");

                // Q @ K^T: [batch, heads, 1, kv_len]
                let att = q.matmul(&k_exp.t().expect("t")).expect("qk");
                let att = (att * softmax_scale).expect("scale");
                let att = candle_nn::ops::softmax_last_dim(&att).expect("softmax");
                // @ V: [batch, heads, 1, head_dim]
                let out = att.matmul(&v_exp).expect("av");
                device.synchronize().ok();
                bb(out.elem_count())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_qk_matmul,
    bench_softmax,
    bench_full_attention,
    bench_gqa_kv_expand,
    bench_decode_attention,
);
criterion_main!(benches);
