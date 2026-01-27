//! Sampling benchmarks for regression detection.
//!
//! Measures token sampling operations:
//! - Top-k filtering (partial sort)
//! - Top-p (nucleus) filtering
//! - Temperature scaling
//! - Combined sampling pipeline
//!
//! Run with: cargo bench --package mistralrs-core --bench sampling

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as bb;

use candle_core::{Device, Result, Tensor, D};

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

/// Vocabulary sizes for different models
const VOCAB_SIZES: &[(&str, usize)] = &[
    ("gpt2", 50257),
    ("llama", 32000),
    ("llama3", 128256),
    ("qwen", 151936),
];

/// Create synthetic logits tensor
fn create_logits(vocab_size: usize, batch_size: usize, device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 10f32, (batch_size, vocab_size), device)
}

// ============================================================================
// Temperature Scaling
// ============================================================================

fn bench_temperature(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_temperature");

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");
        let temperature = 0.7f64;

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                let scaled = (&logits / temperature).expect("scale");
                device.synchronize().ok();
                bb(scaled.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Softmax (for probability conversion)
// ============================================================================

fn bench_softmax(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_softmax");

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                let probs = candle_nn::ops::softmax_last_dim(&logits).expect("softmax");
                device.synchronize().ok();
                bb(probs.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Top-K Filtering (via argsort)
// ============================================================================

fn bench_argsort(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_argsort");
    group.sample_size(30); // Sorting is slow

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                // Full argsort (descending)
                let sorted_indices = logits.arg_sort_last_dim(false).expect("argsort");
                device.synchronize().ok();
                bb(sorted_indices.elem_count())
            });
        });
    }

    group.finish();
}

/// Benchmark: Top-k index selection after sort
fn bench_topk_select(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_topk_select");

    let k_values: &[usize] = &[1, 10, 50, 100];

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");
        // Pre-sort for this benchmark (isolate selection cost)
        let sorted_indices = logits.arg_sort_last_dim(false).expect("argsort");
        let sorted_logits = logits.gather(&sorted_indices, D::Minus1).expect("gather");

        for &k in k_values {
            let id = format!("{}_{}", name, k);

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // Select top-k
                    let top_k_logits = sorted_logits.narrow(D::Minus1, 0, k).expect("narrow");
                    let top_k_indices = sorted_indices.narrow(D::Minus1, 0, k).expect("narrow");
                    device.synchronize().ok();
                    bb(top_k_logits.elem_count() + top_k_indices.elem_count())
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Top-P (Nucleus) Filtering
// ============================================================================

fn bench_cumsum(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_cumsum");

    for (name, vocab_size) in VOCAB_SIZES {
        // Create probability distribution (softmax output)
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");
        let probs = candle_nn::ops::softmax_last_dim(&logits).expect("softmax");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                // Cumulative sum for nucleus sampling
                let cumsum = probs.cumsum(D::Minus1).expect("cumsum");
                device.synchronize().ok();
                bb(cumsum.elem_count())
            });
        });
    }

    group.finish();
}

/// Benchmark: Full top-p filtering pipeline
fn bench_topp_full(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_topp_full");
    group.sample_size(30);

    let top_p = 0.9f32;

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                // 1. Softmax to get probabilities
                let probs = candle_nn::ops::softmax_last_dim(&logits).expect("softmax");
                // 2. Sort probabilities (descending)
                let sorted_indices = probs.arg_sort_last_dim(false).expect("argsort");
                let sorted_probs = probs.gather(&sorted_indices, D::Minus1).expect("gather");
                // 3. Cumulative sum
                let cumsum = sorted_probs.cumsum(D::Minus1).expect("cumsum");
                // 4. Find cutoff (first position where cumsum > top_p)
                // In practice this is done with a threshold mask
                let mask = cumsum.lt(top_p as f64).expect("lt");
                device.synchronize().ok();
                bb(mask.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Min-P Filtering
// ============================================================================

fn bench_minp(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_minp");

    let min_p = 0.05f32;

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");
        let probs = candle_nn::ops::softmax_last_dim(&logits).expect("softmax");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                // Find max probability
                let max_prob = probs.max(D::Minus1).expect("max");
                // Threshold = max_prob * min_p
                let threshold = (max_prob * min_p as f64).expect("mul");
                // Mask tokens below threshold
                let mask = probs
                    .broadcast_ge(&threshold.unsqueeze(D::Minus1).expect("unsqueeze"))
                    .expect("ge");
                device.synchronize().ok();
                bb(mask.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Repetition Penalty
// ============================================================================

fn bench_repetition_penalty(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_repetition_penalty");

    let penalty = 1.1f32;
    let context_lengths: &[usize] = &[64, 256, 1024];

    for (name, vocab_size) in VOCAB_SIZES {
        for &ctx_len in context_lengths {
            let logits = create_logits(*vocab_size, 1, &device).expect("logits");

            // Create token history (random tokens from vocab)
            let history: Vec<u32> = (0..ctx_len).map(|i| (i % *vocab_size) as u32).collect();

            let id = format!("{}_{}", name, ctx_len);

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |b, _| {
                b.iter(|| {
                    // Apply penalty to tokens in history
                    // This is typically done on CPU with scatter
                    let mut logits_vec = logits
                        .flatten_all()
                        .expect("flatten")
                        .to_vec1::<f32>()
                        .expect("to_vec1");

                    for &token_id in &history {
                        let idx = token_id as usize;
                        if logits_vec[idx] > 0.0 {
                            logits_vec[idx] /= penalty;
                        } else {
                            logits_vec[idx] *= penalty;
                        }
                    }

                    // Convert back to tensor
                    let penalized =
                        Tensor::from_vec(logits_vec, *vocab_size, &device).expect("from_vec");
                    bb(penalized.elem_count())
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Full Sampling Pipeline
// ============================================================================

fn bench_full_sampling(c: &mut Criterion) {
    let device = get_device();

    let mut group = c.benchmark_group("sampling_full_pipeline");
    group.sample_size(20);

    let temperature = 0.7f64;
    let top_k = 50usize;
    let top_p = 0.9f32;

    for (name, vocab_size) in VOCAB_SIZES {
        let logits = create_logits(*vocab_size, 1, &device).expect("logits");

        group.throughput(Throughput::Elements(*vocab_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(*name), &(), |b, _| {
            b.iter(|| {
                // 1. Temperature scaling
                let scaled = (&logits / temperature).expect("temp");
                // 2. Softmax
                let probs = candle_nn::ops::softmax_last_dim(&scaled).expect("softmax");
                // 3. Top-k: sort and take top k
                let sorted_indices = probs.arg_sort_last_dim(false).expect("argsort");
                let sorted_probs = probs.gather(&sorted_indices, D::Minus1).expect("gather");
                let top_k_probs = sorted_probs.narrow(D::Minus1, 0, top_k).expect("narrow");
                // 4. Top-p within top-k
                let cumsum = top_k_probs.cumsum(D::Minus1).expect("cumsum");
                let _mask = cumsum.lt(top_p as f64).expect("lt");
                // 5. Would sample here (not benchmarking RNG)
                device.synchronize().ok();
                bb(top_k_probs.elem_count())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_temperature,
    bench_softmax,
    bench_argsort,
    bench_topk_select,
    bench_cumsum,
    bench_topp_full,
    bench_minp,
    bench_repetition_penalty,
    bench_full_sampling,
);
criterion_main!(benches);
