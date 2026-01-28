//! OpenTelemetry metrics for mistral.rs inference server.
//!
//! Implements OTel GenAI semantic conventions plus industry-standard metrics
//! from vLLM and TGI for production observability.
//!
//! ## Metric Categories
//!
//! ### OTel GenAI Standard (Phase 7)
//! - `gen_ai.server.request.duration` - End-to-end request latency
//! - `gen_ai.server.time_to_first_token` - TTFT latency
//! - `gen_ai.server.time_per_output_token` - Inter-token latency
//!
//! ### Industry Standard Extensions (Phase 8)
//! - Latency breakdown: prefill, decode, queue wait
//! - Token distributions: prompt, generation
//! - Counters: requests, tokens, cache hits/misses
//! - Gauges: queue depth, KV cache utilization, batch size

use opentelemetry::{
    metrics::{Counter, Histogram, Meter, MeterProvider},
    KeyValue,
};
use std::sync::OnceLock;

/// Global metrics instance.
static METRICS: OnceLock<MistralRsMetrics> = OnceLock::new();

/// Histogram bucket boundaries for request duration (seconds).
/// Covers 10ms to ~80s with exponential growth.
const REQUEST_DURATION_BUCKETS: &[f64] = &[
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// Histogram bucket boundaries for TTFT (seconds).
/// OTel GenAI recommended boundaries for time-to-first-token.
const TTFT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
];

/// Histogram bucket boundaries for inter-token latency (seconds).
/// OTel GenAI recommended boundaries for time-per-output-token.
const INTER_TOKEN_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
];

/// Histogram bucket boundaries for token counts.
/// Covers typical prompt/generation sizes.
const TOKEN_BUCKETS: &[f64] = &[
    1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
    32768.0,
];

/// OpenTelemetry metrics for mistral.rs.
///
/// This struct holds all metric instruments. Access via [`metrics()`].
#[derive(Debug)]
pub struct MistralRsMetrics {
    // =========================================================================
    // OTel GenAI Standard Metrics (gen_ai.server.*)
    // =========================================================================
    /// End-to-end request duration.
    /// OTel: gen_ai.server.request.duration
    pub request_duration: Histogram<f64>,

    /// Time to first token for streaming responses.
    /// OTel: gen_ai.server.time_to_first_token
    pub time_to_first_token: Histogram<f64>,

    /// Time per output token (inter-token latency).
    /// OTel: gen_ai.server.time_per_output_token
    pub time_per_output_token: Histogram<f64>,

    // =========================================================================
    // Industry Standard Latency Histograms (mistralrs_*)
    // =========================================================================
    /// Prefill/prompt processing duration.
    pub prefill_duration: Histogram<f64>,

    /// Decode/generation duration (excluding TTFT).
    pub decode_duration: Histogram<f64>,

    /// Time spent waiting in scheduler queue.
    pub queue_wait_duration: Histogram<f64>,

    // =========================================================================
    // Token Distribution Histograms
    // =========================================================================
    /// Input/prompt token count distribution.
    pub prompt_tokens: Histogram<u64>,

    /// Output/generation token count distribution.
    pub generation_tokens: Histogram<u64>,

    // =========================================================================
    // Counters
    // =========================================================================
    /// Total requests processed.
    /// Labels: model, operation, status (success/error)
    pub requests_total: Counter<u64>,

    /// Total tokens processed.
    /// Labels: model, type (prompt/generation)
    pub tokens_total: Counter<u64>,

    /// Prefix cache hits.
    pub cache_hits_total: Counter<u64>,

    /// Prefix cache misses.
    pub cache_misses_total: Counter<u64>,
    // =========================================================================
    // Gauges are registered separately via callbacks
    // See register_gauge_callbacks()
    // =========================================================================
}

impl MistralRsMetrics {
    /// Create a new metrics instance from a meter.
    fn new(meter: &Meter) -> Self {
        Self {
            // OTel GenAI standard metrics
            request_duration: meter
                .f64_histogram("gen_ai.server.request.duration")
                .with_description("End-to-end request latency")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS.to_vec())
                .build(),

            time_to_first_token: meter
                .f64_histogram("gen_ai.server.time_to_first_token")
                .with_description("Time to generate first token")
                .with_unit("s")
                .with_boundaries(TTFT_BUCKETS.to_vec())
                .build(),

            time_per_output_token: meter
                .f64_histogram("gen_ai.server.time_per_output_token")
                .with_description("Time per output token after the first")
                .with_unit("s")
                .with_boundaries(INTER_TOKEN_BUCKETS.to_vec())
                .build(),

            // Industry standard latency histograms
            prefill_duration: meter
                .f64_histogram("mistralrs_prefill_duration_seconds")
                .with_description("Prompt/prefill processing duration")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS.to_vec())
                .build(),

            decode_duration: meter
                .f64_histogram("mistralrs_decode_duration_seconds")
                .with_description("Token generation/decode duration")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS.to_vec())
                .build(),

            queue_wait_duration: meter
                .f64_histogram("mistralrs_queue_wait_seconds")
                .with_description("Time spent waiting in scheduler queue")
                .with_unit("s")
                .with_boundaries(TTFT_BUCKETS.to_vec())
                .build(),

            // Token histograms
            prompt_tokens: meter
                .u64_histogram("mistralrs_prompt_tokens")
                .with_description("Input/prompt token count distribution")
                .with_unit("{token}")
                .with_boundaries(TOKEN_BUCKETS.to_vec())
                .build(),

            generation_tokens: meter
                .u64_histogram("mistralrs_generation_tokens")
                .with_description("Output/generation token count distribution")
                .with_unit("{token}")
                .with_boundaries(TOKEN_BUCKETS.to_vec())
                .build(),

            // Counters
            requests_total: meter
                .u64_counter("mistralrs_requests_total")
                .with_description("Total requests processed")
                .with_unit("{request}")
                .build(),

            tokens_total: meter
                .u64_counter("mistralrs_tokens_total")
                .with_description("Total tokens processed")
                .with_unit("{token}")
                .build(),

            cache_hits_total: meter
                .u64_counter("mistralrs_cache_hits_total")
                .with_description("Prefix cache hits")
                .with_unit("{hit}")
                .build(),

            cache_misses_total: meter
                .u64_counter("mistralrs_cache_misses_total")
                .with_description("Prefix cache misses")
                .with_unit("{miss}")
                .build(),
        }
    }

    // =========================================================================
    // Convenience methods for recording metrics with standard labels
    // =========================================================================

    /// Record a completed request.
    ///
    /// This records:
    /// - Request duration histogram
    /// - Request counter
    /// - Token histograms and counters
    pub fn record_request(
        &self,
        duration_secs: f64,
        model: &str,
        operation: &str,
        status: &str,
        prompt_tokens: u64,
        generation_tokens: u64,
    ) {
        let attrs = &[
            KeyValue::new("model", model.to_string()),
            KeyValue::new("gen_ai.operation.name", operation.to_string()),
        ];

        // Duration
        self.request_duration.record(duration_secs, attrs);

        // Counter
        self.requests_total.add(
            1,
            &[
                KeyValue::new("model", model.to_string()),
                KeyValue::new("gen_ai.operation.name", operation.to_string()),
                KeyValue::new("status", status.to_string()),
            ],
        );

        // Token histograms
        self.prompt_tokens.record(prompt_tokens, attrs);
        self.generation_tokens.record(generation_tokens, attrs);

        // Token counters
        self.tokens_total.add(
            prompt_tokens,
            &[
                KeyValue::new("model", model.to_string()),
                KeyValue::new("type", "prompt"),
            ],
        );
        self.tokens_total.add(
            generation_tokens,
            &[
                KeyValue::new("model", model.to_string()),
                KeyValue::new("type", "generation"),
            ],
        );
    }

    /// Record time-to-first-token.
    pub fn record_ttft(&self, ttft_secs: f64, model: &str) {
        self.time_to_first_token
            .record(ttft_secs, &[KeyValue::new("model", model.to_string())]);
    }

    /// Record inter-token latency (time per output token).
    pub fn record_inter_token_latency(&self, latency_secs: f64, model: &str) {
        self.time_per_output_token
            .record(latency_secs, &[KeyValue::new("model", model.to_string())]);
    }

    /// Record prefill duration.
    pub fn record_prefill(&self, duration_secs: f64, model: &str, batch_size: usize) {
        self.prefill_duration.record(
            duration_secs,
            &[
                KeyValue::new("model", model.to_string()),
                KeyValue::new("batch_size", batch_size as i64),
            ],
        );
    }

    /// Record decode duration.
    pub fn record_decode(&self, duration_secs: f64, model: &str, batch_size: usize) {
        self.decode_duration.record(
            duration_secs,
            &[
                KeyValue::new("model", model.to_string()),
                KeyValue::new("batch_size", batch_size as i64),
            ],
        );
    }

    /// Record queue wait time.
    pub fn record_queue_wait(&self, wait_secs: f64, model: &str) {
        self.queue_wait_duration
            .record(wait_secs, &[KeyValue::new("model", model.to_string())]);
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self, model: &str) {
        self.cache_hits_total
            .add(1, &[KeyValue::new("model", model.to_string())]);
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self, model: &str) {
        self.cache_misses_total
            .add(1, &[KeyValue::new("model", model.to_string())]);
    }
}

/// Initialize the global metrics registry.
///
/// This should be called once during server startup, after the meter provider
/// is configured. If called multiple times, subsequent calls are no-ops.
///
/// # Arguments
/// * `meter_provider` - The OTel meter provider (from server-core telemetry setup)
pub fn init_metrics(meter_provider: &dyn MeterProvider) {
    let meter = meter_provider.meter_with_scope(
        opentelemetry::InstrumentationScope::builder("mistralrs")
            .with_version(env!("CARGO_PKG_VERSION"))
            .build(),
    );

    let _ = METRICS.set(MistralRsMetrics::new(&meter));

    tracing::debug!("Metrics registry initialized");
}

/// Get the global metrics instance.
///
/// # Panics
/// Panics if called before [`init_metrics()`]. In practice, this is called
/// after server initialization so this should never happen.
///
/// # Returns
/// Reference to the global metrics registry.
pub fn metrics() -> &'static MistralRsMetrics {
    METRICS
        .get()
        .expect("Metrics not initialized. Call init_metrics() first.")
}

/// Check if metrics have been initialized.
///
/// Useful for conditional metric recording in code paths that may run
/// before server initialization.
pub fn metrics_enabled() -> bool {
    METRICS.get().is_some()
}

/// Try to get metrics, returning None if not initialized.
///
/// Use this in code paths where metrics may not be available (e.g., tests,
/// CLI tools that don't run the server).
pub fn try_metrics() -> Option<&'static MistralRsMetrics> {
    METRICS.get()
}

// =============================================================================
// Gauge Registration
// =============================================================================
//
// Gauges in OTel use callbacks that are invoked on metric collection.
// These must be registered separately because they need access to runtime
// state (scheduler, block engine, etc.).
//
// Example usage in engine initialization:
//
// ```ignore
// use opentelemetry::metrics::MeterProvider;
//
// pub fn register_engine_gauges(meter_provider: &dyn MeterProvider, engine: Arc<Engine>) {
//     let meter = meter_provider.meter("mistralrs");
//     let engine_clone = engine.clone();
//
//     meter
//         .u64_observable_gauge("mistralrs_queue_depth")
//         .with_description("Requests waiting in queue")
//         .with_unit("{request}")
//         .with_callback(move |observer| {
//             observer.observe(engine_clone.queue_len() as u64, &[]);
//         })
//         .build();
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_not_initialized() {
        // metrics() should panic if not initialized
        // We can't easily test this without affecting global state
        assert!(!metrics_enabled());
        assert!(try_metrics().is_none());
    }

    #[test]
    fn test_bucket_boundaries() {
        // Verify bucket arrays are sorted
        assert!(REQUEST_DURATION_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert!(TTFT_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert!(INTER_TOKEN_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert!(TOKEN_BUCKETS.windows(2).all(|w| w[0] < w[1]));
    }
}
