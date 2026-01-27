//! LoRA forward pass benchmarks for regression detection.
//!
//! Measures LoRA adapter operations:
//! - Single adapter forward pass (A*B matmul with scaling)
//! - Multiple adapters (2, 4, 8) forward pass
//! - Stacked vs loop-based adapter application
//! - Tensor stacking for adapter weights
//! - apply_scalings_to_x operation
//! - get_maybe_topk_scalings operation
//!
//! Run with: cargo bench --package mistralrs-core --bench lora_forward
//! Run with Metal: cargo bench --package mistralrs-core --bench lora_forward --features metal

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as bb;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module};

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
// Configuration Constants
// ============================================================================

/// Model hidden dimensions for benchmarking
const HIDDEN_DIMS: &[(&str, usize)] =
    &[("llama_7b", 4096), ("llama_13b", 5120), ("llama_70b", 8192)];

/// Common LoRA ranks
const LORA_RANKS: &[usize] = &[8, 16, 32, 64];

/// Batch sizes for testing
const BATCH_SIZES: &[usize] = &[1, 4, 16];

/// Sequence lengths for testing
const SEQ_LENS: &[usize] = &[128, 512, 2048];

/// Number of adapters for multi-adapter tests
const ADAPTER_COUNTS: &[usize] = &[1, 2, 4, 8];

// ============================================================================
// Helper Functions
// ============================================================================

/// Create synthetic LoRA adapter (A and B matrices)
fn create_lora_adapter(
    in_features: usize,
    out_features: usize,
    rank: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Linear, Linear, f64)> {
    // A: [rank, in_features] - down projection
    let a_weight = Tensor::randn(0f32, 0.01f32, (rank, in_features), device)?.to_dtype(dtype)?;
    let a = Linear::new(a_weight, None);

    // B: [out_features, rank] - up projection
    let b_weight = Tensor::randn(0f32, 0.01f32, (out_features, rank), device)?.to_dtype(dtype)?;
    let b = Linear::new(b_weight, None);

    // Typical scale: alpha / rank (e.g., 32 / 16 = 2.0)
    let scale = 32.0 / rank as f64;

    Ok((a, b, scale))
}

/// Create input tensor for LoRA forward pass
fn create_input(
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_dim), device)?.to_dtype(dtype)
}

/// Create scalings tensor for multi-adapter routing
/// Shape: [batch_size, seq_len, num_layers, num_adapters]
fn create_scalings(
    batch_size: usize,
    seq_len: usize,
    num_layers: usize,
    num_adapters: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // Scalings are typically softmax outputs, so we simulate that
    let raw = Tensor::randn(
        0f32,
        1f32,
        (batch_size, seq_len, num_layers, num_adapters),
        device,
    )?
    .to_dtype(dtype)?;
    candle_nn::ops::softmax_last_dim(&raw)
}

// ============================================================================
// Single Adapter Forward Pass Benchmarks
// ============================================================================

/// Benchmark: Single adapter forward pass (A*B matmul chain with scaling)
fn bench_single_adapter_forward(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_single_adapter_forward");

    for (model_name, hidden_dim) in HIDDEN_DIMS {
        for &rank in LORA_RANKS {
            for &batch_size in &[1usize, 4] {
                for &seq_len in &[128usize, 512] {
                    let (a, b, scale) =
                        create_lora_adapter(*hidden_dim, *hidden_dim, rank, &device, dtype)
                            .expect("adapter");
                    let input = create_input(batch_size, seq_len, *hidden_dim, &device, dtype)
                        .expect("input");

                    let id = format!("{}_r{}_b{}_s{}", model_name, rank, batch_size, seq_len);
                    // FLOPs: 2 * (input @ A^T) + 2 * (intermediate @ B^T)
                    let flops =
                        2 * batch_size * seq_len * (*hidden_dim * rank + rank * *hidden_dim);
                    group.throughput(Throughput::Elements(flops as u64));

                    group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                        bench.iter(|| {
                            // LoRA forward: (input @ A^T @ B^T) * scale
                            let intermediate = a.forward(&input).expect("A forward");
                            let lora_out = b.forward(&intermediate).expect("B forward");
                            let scaled = (lora_out * scale).expect("scale");
                            device.synchronize().ok();
                            bb(scaled.elem_count())
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

/// Benchmark: LoRA delta weight computation (B @ A for merging)
fn bench_delta_weight(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_delta_weight");

    for (model_name, hidden_dim) in HIDDEN_DIMS {
        for &rank in LORA_RANKS {
            let (a, b, scale) = create_lora_adapter(*hidden_dim, *hidden_dim, rank, &device, dtype)
                .expect("adapter");

            let id = format!("{}_r{}", model_name, rank);
            // FLOPs for B @ A: 2 * out_features * rank * in_features
            let flops = 2 * *hidden_dim * rank * *hidden_dim;
            group.throughput(Throughput::Elements(flops as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    // Delta weight: B @ A * scale (for weight merging)
                    let w_a = a.weight();
                    let w_b = b.weight();
                    let delta = w_b.matmul(w_a).expect("matmul");
                    let scaled = (delta * scale).expect("scale");
                    device.synchronize().ok();
                    bb(scaled.elem_count())
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Multiple Adapter Benchmarks
// ============================================================================

/// Benchmark: Loop-based multi-adapter forward (current implementation)
fn bench_multi_adapter_loop(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_multi_adapter_loop");
    group.sample_size(30);

    let hidden_dim = 4096usize; // Llama-7B scale
    let rank = 16usize;
    let batch_size = 1usize;
    let seq_len = 256usize;

    for &num_adapters in ADAPTER_COUNTS {
        // Create multiple adapters
        let adapters: Vec<_> = (0..num_adapters)
            .map(|_| {
                create_lora_adapter(hidden_dim, hidden_dim, rank, &device, dtype).expect("adapter")
            })
            .collect();

        let input = create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");

        let id = format!("adapters_{}", num_adapters);
        let flops =
            2 * num_adapters * batch_size * seq_len * (hidden_dim * rank + rank * hidden_dim);
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
            bench.iter(|| {
                let mut result = Tensor::zeros_like(&input).expect("zeros");
                for (a, b, scale) in &adapters {
                    let intermediate = a.forward(&input).expect("A forward");
                    let lora_out = b.forward(&intermediate).expect("B forward");
                    let scaled = (lora_out * *scale).expect("scale");
                    result = (result + scaled).expect("add");
                }
                device.synchronize().ok();
                bb(result.elem_count())
            });
        });
    }

    group.finish();
}

/// Benchmark: Stacked (batched) multi-adapter forward (optimized path)
fn bench_multi_adapter_stacked(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_multi_adapter_stacked");
    group.sample_size(30);

    let hidden_dim = 4096usize;
    let rank = 16usize;
    let batch_size = 1usize;
    let seq_len = 256usize;

    for &num_adapters in ADAPTER_COUNTS {
        // Create stacked adapter weights: [num_adapters, out, in] for A, [num_adapters, out, in] for B
        let a_weights: Vec<Tensor> = (0..num_adapters)
            .map(|_| {
                Tensor::randn(0f32, 0.01f32, (rank, hidden_dim), &device)
                    .expect("weight")
                    .to_dtype(dtype)
                    .expect("dtype")
            })
            .collect();
        let b_weights: Vec<Tensor> = (0..num_adapters)
            .map(|_| {
                Tensor::randn(0f32, 0.01f32, (hidden_dim, rank), &device)
                    .expect("weight")
                    .to_dtype(dtype)
                    .expect("dtype")
            })
            .collect();

        // Stack into [num_adapters, out, in]
        let a_stacked = Tensor::stack(
            &a_weights
                .iter()
                .map(|w| w.unsqueeze(0).expect("unsqueeze"))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack")
        .squeeze(1)
        .expect("squeeze");

        let b_stacked = Tensor::stack(
            &b_weights
                .iter()
                .map(|w| w.unsqueeze(0).expect("unsqueeze"))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack")
        .squeeze(1)
        .expect("squeeze");

        // Create scales tensor [num_adapters, 1, 1]
        let scales: Vec<f32> = (0..num_adapters).map(|_| 2.0f32).collect();
        let scale_tensor = Tensor::from_vec(scales, (num_adapters, 1, 1), &device).expect("scales");

        // Pre-multiply A weights by scale
        let a_scaled = a_stacked.broadcast_mul(&scale_tensor).expect("scale");

        let input = create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");

        let id = format!("adapters_{}", num_adapters);
        let flops =
            2 * num_adapters * batch_size * seq_len * (hidden_dim * rank + rank * hidden_dim);
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
            bench.iter(|| {
                // Reshape input for batched matmul: [b*s, h]
                let (b, s, h) = input.dims3().expect("dims");
                let input_2d = input.reshape((b * s, h)).expect("reshape");

                // Batched matmul with stacked A: [n_adapters, rank, h] @ [h, b*s] -> [n_adapters, rank, b*s]
                let out_a = a_scaled
                    .broadcast_matmul(&input_2d.t().expect("t"))
                    .expect("matmul A");

                // Batched matmul with stacked B: [n_adapters, h, rank] @ [n_adapters, rank, b*s]
                let out_b = b_stacked.broadcast_matmul(&out_a).expect("matmul B");

                // Sum across adapters and reshape: [n_adapters, h, b*s] -> [b, s, h]
                let out = out_b.sum(0).expect("sum");
                let o_h = out.dims()[0];
                let out = out.t().expect("t").reshape((b, s, o_h)).expect("reshape");

                device.synchronize().ok();
                bb(out.elem_count())
            });
        });
    }

    group.finish();
}

/// Benchmark: Comparison of loop vs stacked at fixed adapter count
fn bench_loop_vs_stacked(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_loop_vs_stacked");
    group.sample_size(50);

    let hidden_dim = 4096usize;
    let rank = 16usize;
    let num_adapters = 4usize;

    // Varying batch and sequence configurations
    let configs = [
        (1, 128, "b1_s128"),
        (1, 512, "b1_s512"),
        (4, 128, "b4_s128"),
        (16, 64, "b16_s64"),
    ];

    for (batch_size, seq_len, config_name) in configs {
        let input = create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");

        // Loop-based adapters
        let adapters: Vec<_> = (0..num_adapters)
            .map(|_| {
                create_lora_adapter(hidden_dim, hidden_dim, rank, &device, dtype).expect("adapter")
            })
            .collect();

        // Stacked adapters
        let a_weights: Vec<Tensor> = adapters
            .iter()
            .map(|(a, _, _)| a.weight().clone())
            .collect();
        let b_weights: Vec<Tensor> = adapters
            .iter()
            .map(|(_, b, _)| b.weight().clone())
            .collect();
        let a_stacked = Tensor::stack(
            &a_weights
                .iter()
                .map(|w| w.unsqueeze(0).expect("unsqueeze"))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack")
        .squeeze(1)
        .expect("squeeze");
        let b_stacked = Tensor::stack(
            &b_weights
                .iter()
                .map(|w| w.unsqueeze(0).expect("unsqueeze"))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack")
        .squeeze(1)
        .expect("squeeze");
        let scales: Vec<f32> = adapters.iter().map(|(_, _, s)| *s as f32).collect();
        let scale_tensor = Tensor::from_vec(scales, (num_adapters, 1, 1), &device).expect("scales");
        let a_scaled = a_stacked.broadcast_mul(&scale_tensor).expect("scale");

        // Benchmark loop-based
        group.bench_with_input(BenchmarkId::new("loop", config_name), &(), |bench, _| {
            bench.iter(|| {
                let mut result = Tensor::zeros_like(&input).expect("zeros");
                for (a, b, scale) in &adapters {
                    let intermediate = a.forward(&input).expect("A forward");
                    let lora_out = b.forward(&intermediate).expect("B forward");
                    let scaled = (lora_out * *scale).expect("scale");
                    result = (result + scaled).expect("add");
                }
                device.synchronize().ok();
                bb(result.elem_count())
            });
        });

        // Benchmark stacked
        group.bench_with_input(BenchmarkId::new("stacked", config_name), &(), |bench, _| {
            bench.iter(|| {
                let (b, s, h) = input.dims3().expect("dims");
                let input_2d = input.reshape((b * s, h)).expect("reshape");
                let out_a = a_scaled
                    .broadcast_matmul(&input_2d.t().expect("t"))
                    .expect("matmul A");
                let out_b = b_stacked.broadcast_matmul(&out_a).expect("matmul B");
                let out = out_b.sum(0).expect("sum");
                let o_h = out.dims()[0];
                let out = out.t().expect("t").reshape((b, s, o_h)).expect("reshape");
                device.synchronize().ok();
                bb(out.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Tensor Operation Benchmarks
// ============================================================================

/// Benchmark: Tensor stacking for adapter weights
fn bench_tensor_stacking(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_tensor_stacking");

    let hidden_dim = 4096usize;

    for &rank in LORA_RANKS {
        for &num_adapters in ADAPTER_COUNTS {
            // Create adapter weights to stack
            let weights: Vec<Tensor> = (0..num_adapters)
                .map(|_| {
                    Tensor::randn(0f32, 0.01f32, (rank, hidden_dim), &device)
                        .expect("weight")
                        .to_dtype(dtype)
                        .expect("dtype")
                })
                .collect();

            let id = format!("r{}_n{}", rank, num_adapters);
            let elements = num_adapters * rank * hidden_dim;
            group.throughput(Throughput::Elements(elements as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    // Stack weights: [n, rank, hidden] via unsqueeze + cat
                    let unsqueezed: Vec<Tensor> = weights
                        .iter()
                        .map(|w| w.unsqueeze(0).expect("unsqueeze"))
                        .collect();
                    let stacked = Tensor::cat(&unsqueezed, 0).expect("cat");
                    device.synchronize().ok();
                    bb(stacked.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: apply_scalings_to_x operation
/// Mirrors the function from mistralrs-core/src/lora/mod.rs
fn bench_apply_scalings_to_x(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_apply_scalings_to_x");

    let hidden_dim = 4096usize;
    let num_layers = 32usize;
    let num_adapters = 4usize;

    for &batch_size in BATCH_SIZES {
        for &seq_len in &[128usize, 512, 1024] {
            let input =
                create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");
            let scalings = create_scalings(
                batch_size,
                seq_len,
                num_layers,
                num_adapters,
                &device,
                dtype,
            )
            .expect("scalings");

            let id = format!("b{}_s{}", batch_size, seq_len);
            let elements = batch_size * seq_len * hidden_dim;
            group.throughput(Throughput::Elements(elements as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    // apply_scalings_to_x: x * scalings[..., adapter].unsqueeze(-1)
                    let adapter_idx = 0usize;
                    let layer_idx = 0usize;

                    // First get the layer scalings: [batch, seq, num_adapters]
                    let layer_scalings = scalings.i((.., .., layer_idx, ..)).expect("layer index");

                    // Then get specific adapter: [batch, seq]
                    let adapter_scalings = layer_scalings
                        .i((.., .., adapter_idx))
                        .expect("adapter index");

                    // Unsqueeze and broadcast multiply
                    let scaling_expanded =
                        adapter_scalings.unsqueeze(D::Minus1).expect("unsqueeze");
                    let result = input.broadcast_mul(&scaling_expanded).expect("mul");

                    device.synchronize().ok();
                    bb(result.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: get_maybe_topk_scalings operation
/// Mirrors the function from mistralrs-core/src/lora/mod.rs
fn bench_get_maybe_topk_scalings(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_get_maybe_topk_scalings");

    let num_layers = 32usize;
    let num_adapters = 4usize;

    for &batch_size in BATCH_SIZES {
        for &seq_len in &[128usize, 512, 2048] {
            let scalings = create_scalings(
                batch_size,
                seq_len,
                num_layers,
                num_adapters,
                &device,
                dtype,
            )
            .expect("scalings");

            let id = format!("b{}_s{}", batch_size, seq_len);
            let elements = batch_size * seq_len * num_adapters;
            group.throughput(Throughput::Elements(elements as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    // get_maybe_topk_scalings: scalings[..., layer, ...]
                    let layer_idx = 16usize; // Middle layer
                    let result = scalings.i((.., .., layer_idx, ..)).expect("index");
                    device.synchronize().ok();
                    bb(result.elem_count())
                });
            });
        }
    }

    group.finish();
}

/// Benchmark: Scaling multiplication patterns
fn bench_scaling_patterns(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_scaling_patterns");

    let hidden_dim = 4096usize;
    let batch_size = 1usize;
    let seq_len = 512usize;

    let input = create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");

    // Different scaling patterns
    let scalar_scale = 2.0f64;
    let per_token_scale = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 1), &device)
        .expect("scale")
        .to_dtype(dtype)
        .expect("dtype");
    let per_element_scale = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_dim), &device)
        .expect("scale")
        .to_dtype(dtype)
        .expect("dtype");

    // Scalar multiplication
    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            let result = (&input * scalar_scale).expect("mul");
            device.synchronize().ok();
            bb(result.elem_count())
        });
    });

    // Per-token broadcast multiplication
    group.bench_function("per_token_broadcast", |bench| {
        bench.iter(|| {
            let result = input.broadcast_mul(&per_token_scale).expect("mul");
            device.synchronize().ok();
            bb(result.elem_count())
        });
    });

    // Per-element multiplication
    group.bench_function("per_element", |bench| {
        bench.iter(|| {
            let result = (&input * &per_element_scale).expect("mul");
            device.synchronize().ok();
            bb(result.elem_count())
        });
    });

    group.finish();
}

// ============================================================================
// Routing Overhead Benchmarks
// ============================================================================

/// Benchmark: Scalings softmax computation (adapter routing)
fn bench_routing_softmax(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_routing_softmax");

    let num_layers = 32usize;

    for &num_adapters in ADAPTER_COUNTS {
        for &batch_size in BATCH_SIZES {
            for &seq_len in &[128usize, 512] {
                // Raw routing logits
                let logits = Tensor::randn(
                    0f32,
                    1f32,
                    (batch_size, seq_len, num_layers, num_adapters),
                    &device,
                )
                .expect("logits")
                .to_dtype(dtype)
                .expect("dtype");

                let id = format!("a{}_b{}_s{}", num_adapters, batch_size, seq_len);
                let elements = batch_size * seq_len * num_layers * num_adapters;
                group.throughput(Throughput::Elements(elements as u64));

                group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                    bench.iter(|| {
                        let scalings = candle_nn::ops::softmax_last_dim(&logits).expect("softmax");
                        device.synchronize().ok();
                        bb(scalings.elem_count())
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark: Top-k selection for adapter routing
fn bench_routing_topk(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_routing_topk");

    let num_layers = 32usize;
    let num_adapters = 8usize;
    let batch_size = 1usize;
    let seq_len = 512usize;
    let top_k_values = [1usize, 2, 4];

    // Scalings tensor (after softmax)
    let scalings = create_scalings(
        batch_size,
        seq_len,
        num_layers,
        num_adapters,
        &device,
        dtype,
    )
    .expect("scalings");

    for &k in &top_k_values {
        let id = format!("topk_{}", k);

        group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
            bench.iter(|| {
                // Top-k selection across adapters (last dim)
                // Reshape for argsort: [batch * seq * layers, adapters]
                let (b, s, l, a) = scalings.dims4().expect("dims");
                let flat = scalings.reshape((b * s * l, a)).expect("reshape");

                // Argsort (descending) and take top-k
                let sorted_indices = flat.arg_sort_last_dim(false).expect("argsort");
                let topk_indices = sorted_indices.narrow(D::Minus1, 0, k).expect("narrow");
                let topk_values = flat.gather(&topk_indices, D::Minus1).expect("gather");

                device.synchronize().ok();
                bb(topk_values.elem_count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Combined Forward Pass Benchmarks
// ============================================================================

/// Benchmark: Full LoRA forward pass (base + adapters)
fn bench_full_lora_forward(c: &mut Criterion) {
    let device = get_device();
    let dtype = DType::F32;

    let mut group = c.benchmark_group("lora_full_forward");
    group.sample_size(30);

    let hidden_dim = 4096usize;
    let rank = 16usize;
    let num_adapters = 4usize;

    for &batch_size in &[1usize, 4] {
        for &seq_len in &[128usize, 512] {
            // Base linear layer
            let base_weight = Tensor::randn(0f32, 0.01f32, (hidden_dim, hidden_dim), &device)
                .expect("weight")
                .to_dtype(dtype)
                .expect("dtype");
            let base_bias = Tensor::randn(0f32, 0.01f32, (hidden_dim,), &device)
                .expect("bias")
                .to_dtype(dtype)
                .expect("dtype");
            let base = Linear::new(base_weight, Some(base_bias));

            // LoRA adapters
            let adapters: Vec<_> = (0..num_adapters)
                .map(|_| {
                    create_lora_adapter(hidden_dim, hidden_dim, rank, &device, dtype)
                        .expect("adapter")
                })
                .collect();

            let input =
                create_input(batch_size, seq_len, hidden_dim, &device, dtype).expect("input");
            let global_scaling_weight = 1.0f64;

            let id = format!("b{}_s{}", batch_size, seq_len);
            // FLOPs: base linear + all adapters
            let base_flops = 2 * batch_size * seq_len * hidden_dim * hidden_dim;
            let lora_flops =
                num_adapters * 2 * batch_size * seq_len * (hidden_dim * rank + rank * hidden_dim);
            group.throughput(Throughput::Elements((base_flops + lora_flops) as u64));

            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    // Base layer forward
                    let mut result = base.forward(&input).expect("base forward");

                    // Add LoRA contributions
                    for (a, b, scale) in &adapters {
                        let intermediate = a.forward(&input).expect("A forward");
                        let lora_out = b.forward(&intermediate).expect("B forward");
                        let scaled = (lora_out * (*scale * global_scaling_weight)).expect("scale");
                        result = (result + scaled).expect("add");
                    }

                    device.synchronize().ok();
                    bb(result.elem_count())
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    // Single adapter operations
    bench_single_adapter_forward,
    bench_delta_weight,
    // Multi-adapter operations
    bench_multi_adapter_loop,
    bench_multi_adapter_stacked,
    bench_loop_vs_stacked,
    // Tensor operations
    bench_tensor_stacking,
    bench_apply_scalings_to_x,
    bench_get_maybe_topk_scalings,
    bench_scaling_patterns,
    // Routing operations
    bench_routing_softmax,
    bench_routing_topk,
    // Combined forward pass
    bench_full_lora_forward,
);
criterion_main!(benches);
