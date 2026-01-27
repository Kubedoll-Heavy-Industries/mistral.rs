//! Layer primitive benchmarks for regression detection.
//!
//! Measures core layer operations:
//! - RmsNorm (called 2x per block per token)
//! - RoPE (rotary positional embedding)
//! - MLP forward pass (SwiGLU pattern)
//!
//! Run with: cargo bench --package mistralrs-core --bench layers
//! Run with Metal: cargo bench --package mistralrs-core --bench layers --features metal

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as bb;

use candle_core::{DType, Device, Result, Tensor};

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

// ============================================================================
// RmsNorm Benchmarks
// ============================================================================

/// Model hidden dimensions for benchmarking
const HIDDEN_SIZES: &[(&str, usize)] = &[
    ("qwen_0.6b", 896),
    ("llama_1b", 2048),
    ("llama_7b", 4096),
    ("llama_13b", 5120),
    ("llama_70b", 8192),
];

const SEQ_LENS: &[usize] = &[1, 64, 256, 1024];

/// Benchmark: RmsNorm forward pass
fn bench_rmsnorm(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;
    let eps = 1e-5f64;

    let mut group = c.benchmark_group("rmsnorm");

    for (model_name, hidden_size) in HIDDEN_SIZES {
        for &seq_len in SEQ_LENS {
            // Input: [batch=1, seq_len, hidden_size]
            let x = Tensor::randn(0f32, 1f32, (1, seq_len, *hidden_size), &device)
                .expect("tensor")
                .to_dtype(dtype)
                .expect("dtype");
            let weight = Tensor::randn(0f32, 1f32, (*hidden_size,), &device)
                .expect("weight")
                .to_dtype(dtype)
                .expect("dtype");

            let id = format!("{}_{}", model_name, seq_len);
            group.throughput(Throughput::Elements((seq_len * hidden_size) as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    let result =
                        candle_nn::ops::rms_norm(&x, &weight, eps as f32).expect("rmsnorm");
                    device.synchronize().ok();
                    bb(result.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: RmsNorm with contiguous() call (common pattern)
fn bench_rmsnorm_contiguous(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;
    let eps = 1e-5f64;

    let mut group = c.benchmark_group("rmsnorm_contiguous");

    for (model_name, hidden_size) in HIDDEN_SIZES {
        // Non-contiguous tensor (from transpose)
        let x = Tensor::randn(0f32, 1f32, (*hidden_size, 256), &device)
            .expect("tensor")
            .to_dtype(dtype)
            .expect("dtype")
            .t()
            .expect("transpose"); // Now [256, hidden_size] but non-contiguous

        let weight = Tensor::randn(0f32, 1f32, (*hidden_size,), &device)
            .expect("weight")
            .to_dtype(dtype)
            .expect("dtype");

        group.throughput(Throughput::Elements((256 * hidden_size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*model_name), &(), |b, _| {
            b.iter(|| {
                let x_contig = x.contiguous().expect("contiguous");
                let result =
                    candle_nn::ops::rms_norm(&x_contig, &weight, eps as f32).expect("rmsnorm");
                device.synchronize().ok();
                bb(result.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// RoPE Benchmarks
// ============================================================================

/// RoPE configurations
struct RopeConfig {
    name: &'static str,
    head_dim: usize,
    num_heads: usize,
    max_position: usize,
}

const ROPE_CONFIGS: &[RopeConfig] = &[
    RopeConfig {
        name: "llama_7b",
        head_dim: 128,
        num_heads: 32,
        max_position: 4096,
    },
    RopeConfig {
        name: "mistral_7b",
        head_dim: 128,
        num_heads: 32,
        max_position: 8192,
    },
    RopeConfig {
        name: "qwen_0.6b",
        head_dim: 64,
        num_heads: 14,
        max_position: 4096,
    },
];

/// Precompute RoPE cos/sin tables
fn create_rope_tables(
    head_dim: usize,
    max_pos: usize,
    base: f32,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / base.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

    let t = Tensor::arange(0u32, max_pos as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_pos, 1))?;
    let freqs = t.matmul(&inv_freq)?;

    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;

    Ok((cos, sin))
}

/// Benchmark: RoPE table lookup (narrowing cos/sin for position)
fn bench_rope_lookup(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("rope_lookup");

    for config in ROPE_CONFIGS {
        let (cos, sin) = create_rope_tables(
            config.head_dim,
            config.max_position,
            10000.0,
            &device,
            dtype,
        )
        .expect("rope tables");

        for &seq_len in &[1usize, 64, 256, 1024] {
            let offset = 100usize; // Typical decode position

            let id = format!("{}_{}", config.name, seq_len);

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    let cos_slice = cos.narrow(0, offset, seq_len).expect("narrow cos");
                    let sin_slice = sin.narrow(0, offset, seq_len).expect("narrow sin");
                    device.synchronize().ok();
                    bb(cos_slice.elem_count() + sin_slice.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: RoPE application (GPT-NeoX style rotation)
fn bench_rope_apply(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("rope_apply");
    group.sample_size(50);

    for config in ROPE_CONFIGS {
        let (cos, sin) = create_rope_tables(
            config.head_dim,
            config.max_position,
            10000.0,
            &device,
            dtype,
        )
        .expect("rope tables");

        for &seq_len in &[1usize, 64, 256] {
            // Q tensor: [batch, heads, seq_len, head_dim]
            let q = Tensor::randn(
                0f32,
                1f32,
                (1, config.num_heads, seq_len, config.head_dim),
                &device,
            )
            .expect("q")
            .to_dtype(dtype)
            .expect("dtype");
            let k = Tensor::randn(
                0f32,
                1f32,
                (1, config.num_heads, seq_len, config.head_dim),
                &device,
            )
            .expect("k")
            .to_dtype(dtype)
            .expect("dtype");

            let cos_slice = cos.narrow(0, 0, seq_len).expect("narrow cos");
            let sin_slice = sin.narrow(0, 0, seq_len).expect("narrow sin");

            let id = format!("{}_{}", config.name, seq_len);
            let elements = 2 * config.num_heads * seq_len * config.head_dim;
            group.throughput(Throughput::Elements(elements as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // Apply RoPE using candle_nn (GPT-NeoX style)
                    let q_embed = candle_nn::rotary_emb::rope(
                        &q.contiguous().expect("contig"),
                        &cos_slice,
                        &sin_slice,
                    )
                    .expect("rope q");
                    let k_embed = candle_nn::rotary_emb::rope(
                        &k.contiguous().expect("contig"),
                        &cos_slice,
                        &sin_slice,
                    )
                    .expect("rope k");
                    device.synchronize().ok();
                    bb(q_embed.elem_count() + k_embed.elem_count())
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// MLP Benchmarks
// ============================================================================

/// MLP configurations (SwiGLU pattern)
struct MlpConfig {
    name: &'static str,
    hidden_size: usize,
    intermediate_size: usize, // Usually 4x or ~2.7x hidden
}

const MLP_CONFIGS: &[MlpConfig] = &[
    MlpConfig {
        name: "qwen_0.6b",
        hidden_size: 896,
        intermediate_size: 4864,
    },
    MlpConfig {
        name: "llama_7b",
        hidden_size: 4096,
        intermediate_size: 11008,
    },
    MlpConfig {
        name: "mistral_7b",
        hidden_size: 4096,
        intermediate_size: 14336,
    },
];

/// Benchmark: MLP gate projection (hidden -> intermediate)
fn bench_mlp_projection(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("mlp_projection");
    group.sample_size(30);

    for config in MLP_CONFIGS {
        for &seq_len in &[1usize, 64, 256] {
            // Input: [batch, seq_len, hidden_size]
            let x = Tensor::randn(0f32, 1f32, (1, seq_len, config.hidden_size), &device)
                .expect("x")
                .to_dtype(dtype)
                .expect("dtype");

            // Weight: [intermediate_size, hidden_size]
            let w = Tensor::randn(
                0f32,
                1f32,
                (config.intermediate_size, config.hidden_size),
                &device,
            )
            .expect("w")
            .to_dtype(dtype)
            .expect("dtype");

            let id = format!("{}_{}_gate", config.name, seq_len);
            let flops = 2 * seq_len * config.hidden_size * config.intermediate_size;
            group.throughput(Throughput::Elements(flops as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // x @ w^T
                    let out = x.matmul(&w.t().expect("t")).expect("matmul");
                    device.synchronize().ok();
                    bb(out.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: SwiGLU activation (gate * silu(up))
fn bench_swiglu(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("mlp_swiglu");

    for config in MLP_CONFIGS {
        for &seq_len in &[1usize, 64, 256] {
            // After gate and up projections: [batch, seq_len, intermediate_size]
            let gate = Tensor::randn(0f32, 1f32, (1, seq_len, config.intermediate_size), &device)
                .expect("gate")
                .to_dtype(dtype)
                .expect("dtype");
            let up = Tensor::randn(0f32, 1f32, (1, seq_len, config.intermediate_size), &device)
                .expect("up")
                .to_dtype(dtype)
                .expect("dtype");

            let id = format!("{}_{}", config.name, seq_len);
            group.throughput(Throughput::Elements(
                (seq_len * config.intermediate_size) as u64,
            ));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // SwiGLU: gate * silu(up)
                    let silu_up = candle_nn::ops::silu(&up).expect("silu");
                    let out = (gate.clone() * silu_up).expect("mul");
                    device.synchronize().ok();
                    bb(out.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: Full MLP forward (gate + up projections, SwiGLU, down projection)
fn bench_mlp_full(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("mlp_full");
    group.sample_size(20);

    for config in MLP_CONFIGS {
        for &seq_len in &[1usize, 64, 256] {
            let x = Tensor::randn(0f32, 1f32, (1, seq_len, config.hidden_size), &device)
                .expect("x")
                .to_dtype(dtype)
                .expect("dtype");

            // Weights
            let w_gate = Tensor::randn(
                0f32,
                1f32,
                (config.intermediate_size, config.hidden_size),
                &device,
            )
            .expect("w_gate")
            .to_dtype(dtype)
            .expect("dtype");
            let w_up = Tensor::randn(
                0f32,
                1f32,
                (config.intermediate_size, config.hidden_size),
                &device,
            )
            .expect("w_up")
            .to_dtype(dtype)
            .expect("dtype");
            let w_down = Tensor::randn(
                0f32,
                1f32,
                (config.hidden_size, config.intermediate_size),
                &device,
            )
            .expect("w_down")
            .to_dtype(dtype)
            .expect("dtype");

            let id = format!("{}_{}", config.name, seq_len);
            // FLOPs: 2 * (gate + up + down) projections
            let flops = 2
                * seq_len
                * (2 * config.hidden_size * config.intermediate_size
                    + config.intermediate_size * config.hidden_size);
            group.throughput(Throughput::Elements(flops as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // Gate and up projections
                    let gate = x.matmul(&w_gate.t().expect("t")).expect("gate");
                    let up = x.matmul(&w_up.t().expect("t")).expect("up");
                    // SwiGLU
                    let silu_up = candle_nn::ops::silu(&up).expect("silu");
                    let hidden = (gate * silu_up).expect("mul");
                    // Down projection
                    let out = hidden.matmul(&w_down.t().expect("t")).expect("down");
                    device.synchronize().ok();
                    bb(out.elem_count())
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rmsnorm,
    bench_rmsnorm_contiguous,
    bench_rope_lookup,
    bench_rope_apply,
    bench_mlp_projection,
    bench_swiglu,
    bench_mlp_full,
);
criterion_main!(benches);
