//! Telemetry infrastructure for mistral.rs.
//!
//! This module provides OpenTelemetry metrics for production observability.
//! It follows OTel GenAI semantic conventions and industry best practices
//! from vLLM and TGI.
//!
//! ## Architecture
//!
//! Metrics are defined here in `mistralrs-core` so they can be recorded from:
//! - Engine loop (queue depth, batch size, forward pass timing)
//! - Scheduler (queue wait time)
//! - Cache systems (KV cache utilization, prefix cache hits)
//! - Server endpoints (request duration, TTFT, inter-token latency)
//!
//! The meter provider is initialized by `mistralrs-server-core` which sets up
//! the OTLP exporter. This module provides the metrics registry that records
//! to that provider.
//!
//! ## Usage
//!
//! ```ignore
//! use mistralrs_core::telemetry::metrics;
//!
//! // Record request duration
//! metrics().request_duration.record(1.5, &[
//!     KeyValue::new("model", "llama-7b"),
//!     KeyValue::new("operation", "chat"),
//! ]);
//!
//! // Increment request counter
//! metrics().requests_total.add(1, &[
//!     KeyValue::new("status", "success"),
//! ]);
//! ```

mod metrics;

pub use metrics::{init_metrics, metrics, metrics_enabled, try_metrics, MistralRsMetrics};
