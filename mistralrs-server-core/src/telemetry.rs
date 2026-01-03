//! OpenTelemetry and OpenInference telemetry integration.
//!
//! This module provides:
//! - OTLP exporter configuration
//! - Trace context extraction from HTTP headers
//! - OpenInference span builders for LLM operations

use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::{Sampler, SdkTracerProvider},
    Resource,
};
use std::time::Duration;

/// Configuration for telemetry export.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: Option<String>,
    /// Service name for span metadata
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Whether to record message content (may contain sensitive data)
    pub record_content: bool,
    /// Sampling ratio (0.0 to 1.0)
    pub sampling_ratio: f64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
            service_name: std::env::var("OTEL_SERVICE_NAME")
                .unwrap_or_else(|_| "mistral-rs".to_string()),
            service_version: std::env::var("OTEL_SERVICE_VERSION")
                .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string()),
            record_content: std::env::var("OPENINFERENCE_HIDE_INPUTS")
                .map(|v| v != "true")
                .unwrap_or(true)
                && std::env::var("OPENINFERENCE_HIDE_OUTPUTS")
                    .map(|v| v != "true")
                    .unwrap_or(true),
            sampling_ratio: std::env::var("OTEL_TRACES_SAMPLER_ARG")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
        }
    }
}

/// Initialize the OpenTelemetry tracer provider with OTLP export.
///
/// Returns `None` if no OTLP endpoint is configured.
pub fn init_telemetry(config: &TelemetryConfig) -> Option<SdkTracerProvider> {
    let endpoint = config.otlp_endpoint.as_ref()?;

    // Set up trace context propagation (W3C format)
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Build the OTLP exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create OTLP exporter");

    // Build the tracer provider
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(Sampler::TraceIdRatioBased(config.sampling_ratio))
        .with_resource(Resource::builder().with_attributes(vec![
            KeyValue::new("service.name", config.service_name.clone()),
            KeyValue::new("service.version", config.service_version.clone()),
            KeyValue::new("telemetry.sdk.language", "rust"),
            KeyValue::new("telemetry.sdk.name", "opentelemetry"),
        ]).build())
        .build();

    global::set_tracer_provider(provider.clone());

    tracing::info!(
        endpoint = %endpoint,
        service = %config.service_name,
        "OpenTelemetry OTLP exporter initialized"
    );

    Some(provider)
}

/// Shutdown telemetry and flush pending spans.
pub fn shutdown_telemetry(provider: SdkTracerProvider) {
    let _ = provider.shutdown();
}

/// Extract trace context from HTTP headers.
///
/// Supports W3C Trace Context (traceparent/tracestate headers).
pub fn extract_trace_context(
    headers: &axum::http::HeaderMap,
) -> opentelemetry::Context {
    use opentelemetry::propagation::TextMapPropagator;

    struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);

    impl<'a> opentelemetry::propagation::Extractor for HeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }

        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }

    let propagator = TraceContextPropagator::new();
    propagator.extract(&HeaderExtractor(headers))
}

/// Create a tracing span with OpenInference LLM attributes.
///
/// This is a convenience macro that creates a span with the correct
/// OpenInference attributes for LLM operations.
#[macro_export]
macro_rules! llm_span {
    ($model:expr) => {
        tracing::info_span!(
            "llm",
            otel.name = format!("llm {}", $model),
            "openinference.span.kind" = "LLM",
            "llm.model_name" = %$model,
            "llm.provider" = "mistral.rs",
            "gen_ai.request.model" = %$model,
            "gen_ai.provider.name" = "mistral.rs",
            // These will be filled in later with span.record()
            "llm.token_count.prompt" = tracing::field::Empty,
            "llm.token_count.completion" = tracing::field::Empty,
            "llm.token_count.total" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
        )
    };
    ($model:expr, $($field:tt)*) => {
        tracing::info_span!(
            "llm",
            otel.name = format!("llm {}", $model),
            "openinference.span.kind" = "LLM",
            "llm.model_name" = %$model,
            "llm.provider" = "mistral.rs",
            "gen_ai.request.model" = %$model,
            "gen_ai.provider.name" = "mistral.rs",
            "llm.token_count.prompt" = tracing::field::Empty,
            "llm.token_count.completion" = tracing::field::Empty,
            "llm.token_count.total" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
            $($field)*
        )
    };
}

/// Create a tracing span for embedding operations.
#[macro_export]
macro_rules! embedding_span {
    ($model:expr) => {
        tracing::info_span!(
            "embedding",
            otel.name = format!("embedding {}", $model),
            "openinference.span.kind" = "EMBEDDING",
            "embedding.model_name" = %$model,
            "llm.provider" = "mistral.rs",
            "embedding.token_count.total" = tracing::field::Empty,
        )
    };
}

/// Record token usage on a span.
pub fn record_token_usage(
    span: &tracing::Span,
    prompt_tokens: usize,
    completion_tokens: usize,
) {
    let total = prompt_tokens + completion_tokens;

    // OpenInference attributes
    span.record("llm.token_count.prompt", prompt_tokens as i64);
    span.record("llm.token_count.completion", completion_tokens as i64);
    span.record("llm.token_count.total", total as i64);

    // OTel GenAI attributes
    span.record("gen_ai.usage.input_tokens", prompt_tokens as i64);
    span.record("gen_ai.usage.output_tokens", completion_tokens as i64);
}

/// Record timing metrics on a span.
#[allow(dead_code)]
pub fn record_timing(
    span: &tracing::Span,
    prompt_time_sec: f32,
    completion_time_sec: f32,
    prompt_throughput: f32,
    completion_throughput: f32,
) {
    span.record("llm.timing.prompt_time_sec", prompt_time_sec as f64);
    span.record("llm.timing.completion_time_sec", completion_time_sec as f64);
    span.record(
        "llm.timing.prompt_throughput_tok_per_sec",
        prompt_throughput as f64,
    );
    span.record(
        "llm.timing.completion_throughput_tok_per_sec",
        completion_throughput as f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::TraceContextExt;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "mistral-rs");
        assert!(config.sampling_ratio > 0.0);
    }

    #[test]
    fn test_extract_empty_headers() {
        let headers = axum::http::HeaderMap::new();
        let ctx = extract_trace_context(&headers);
        // Should return a valid (but empty) context
        assert!(!ctx.span().span_context().is_valid());
    }
}
