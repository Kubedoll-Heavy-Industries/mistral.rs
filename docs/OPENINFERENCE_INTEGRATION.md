# OpenInference/OpenLLMetry Integration Guide

This document provides a comprehensive guide for integrating OpenInference semantic conventions and OpenTelemetry-based observability into mistral.rs.

## Table of Contents

1. [Overview of OpenInference and OpenLLMetry](#overview)
2. [Current Telemetry Implementation](#current-implementation)
3. [Integration Points in mistral.rs](#integration-points)
4. [Recommended Instrumentation Strategy](#recommended-strategy)
5. [Code Examples](#code-examples)
6. [References](#references)

---

## Overview

### What is OpenInference?

[OpenInference](https://github.com/Arize-ai/openinference) is a set of semantic conventions and plugins complementary to OpenTelemetry designed for tracing AI/LLM applications. It provides standardized attributes for:

- **LLM spans**: Model calls, token usage, prompts, completions
- **Embedding spans**: Embedding model operations
- **Retriever spans**: RAG/vector store operations
- **Reranker spans**: Cross-encoder reranking operations
- **Tool/Agent spans**: Function calling and agent workflows
- **Guardrail spans**: Content moderation and safety checks

### What is OpenLLMetry?

[OpenLLMetry](https://github.com/traceloop/openllmetry) (Open Large Language Model Telemetry) extends OpenTelemetry with LLM-specific instrumentation:

- Automatic capture of model name/version
- Token counts (prompt, completion, total)
- Temperature and other inference parameters
- Latency metrics and time-to-first-token (TTFT)
- Prompt and completion content (with optional redaction)

### Key Semantic Conventions

#### OpenInference Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `openinference.span.kind` | string | **Required**. Span type: LLM, EMBEDDING, RETRIEVER, RERANKER, TOOL, AGENT, CHAIN, GUARDRAIL |
| `llm.model_name` | string | Model identifier (e.g., "llama-3.2-1b-instruct") |
| `llm.provider` | string | Provider name (e.g., "mistral.rs") |
| `llm.invocation_parameters` | string | JSON-encoded inference parameters |
| `llm.token_count.prompt` | int | Number of prompt/input tokens |
| `llm.token_count.completion` | int | Number of completion/output tokens |
| `llm.token_count.total` | int | Total tokens (prompt + completion) |
| `llm.input_messages.<i>.message.role` | string | Message role at index i |
| `llm.input_messages.<i>.message.content` | string | Message content at index i |
| `llm.output_messages.<i>.message.role` | string | Output message role |
| `llm.output_messages.<i>.message.content` | string | Output message content |
| `llm.prompt_template.template` | string | Prompt template used |
| `llm.prompt_template.variables` | string | Template variable values |

#### OpenTelemetry GenAI Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.request.model` | string | Requested model name |
| `gen_ai.response.model` | string | Actual model used |
| `gen_ai.provider.name` | string | Provider identifier |
| `gen_ai.usage.input_tokens` | int | Input token count |
| `gen_ai.usage.output_tokens` | int | Output token count |
| `gen_ai.request.max_tokens` | int | Max tokens requested |
| `gen_ai.request.temperature` | float | Temperature setting |

#### Cache-Related Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm.token_count.prompt_details.cache_read` | int | Tokens read from cache (cache hits) |
| `llm.token_count.prompt_details.cache_write` | int | Tokens written to cache |

---

## Current Implementation

mistral.rs already has a foundation for telemetry in `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/telemetry.rs`:

### Existing Features

1. **OTLP Exporter Configuration**: Support for exporting traces to any OpenTelemetry-compatible backend
2. **W3C Trace Context Propagation**: Extract trace context from HTTP headers
3. **OpenInference Span Macros**: `llm_span!` and `embedding_span!` macros for creating instrumented spans
4. **Token Usage Recording**: `record_token_usage()` helper function

### Current Dependencies (from Cargo.toml)

```toml
tracing-opentelemetry = "0.32"
opentelemetry = "0.31"
opentelemetry_sdk = { version = "0.31", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.31", features = ["grpc-tonic"] }
openinference-semantic-conventions = { git = "..." }
openinference-instrumentation = { git = "..." }
```

### Current Instrumentation Points

Located in `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/`:

1. **Chat Completions** (`chat_completion.rs:593-601`):
   ```rust
   #[tracing::instrument(
       name = "chat_completion",
       skip(state, oairequest),
       fields(
           otel.kind = "server",
           llm.model_name = %oairequest.model,
           openinference.span.kind = "LLM"
       )
   )]
   pub async fn chatcompletions(...) -> ChatCompletionResponder
   ```

2. **Embeddings** (`embeddings.rs:65-73`):
   ```rust
   #[tracing::instrument(
       name = "embeddings",
       skip(state, oairequest),
       fields(
           otel.kind = "server",
           llm.model_name = %oairequest.model,
           openinference.span.kind = "EMBEDDING"
       )
   )]
   pub async fn embeddings(...) -> EmbeddingResponder
   ```

3. **Reranking** (`reranking.rs:55-63`):
   ```rust
   #[tracing::instrument(
       name = "rerank",
       skip(state, request),
       fields(
           otel.kind = "server",
           llm.model_name = %request.model,
           openinference.span.kind = "RERANKER"
       )
   )]
   pub async fn rerank(...) -> RerankResponder
   ```

---

## Integration Points in mistral.rs

### High-Priority Integration Points

#### 1. Chat Completion Handler (Server Layer)

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/chat_completion.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 602-640 | `chatcompletions()` entry | Request parameters, streaming flag |
| Line 627-636 | Non-streaming response | Token counts recorded (already done) |
| Line 624-625 | Streaming response | TTFT event, per-chunk events |

**Current gap**: Streaming responses don't record token usage on span completion.

#### 2. Completions Handler

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/completions.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 282-302 | `completions()` entry | Add `#[tracing::instrument]` with OpenInference attributes |
| Line 300-301 | Non-streaming response | Record token usage |

**Current gap**: Missing `#[tracing::instrument]` decoration.

#### 3. Embeddings Handler

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/embeddings.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 74-218 | `embeddings()` function | Record `embedding.token_count.total` |

**Current gap**: Token count not recorded on span.

#### 4. Reranking Handler

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-server-core/src/reranking.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 64-95 | `rerank()` function | Record token usage from response |

**Current gap**: Token count not recorded on span.

### Medium-Priority Integration Points

#### 5. Engine Forward Pass

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-core/src/engine/mod.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 399-759 | Main engine loop | Prompt/completion timing, batch sizes |
| Line 437-448 | Completion step | Completion token count, throughput |
| Line 484-600 | Prompt step | Prompt processing time, TTFT |

**Recommended**: Add spans around forward passes for detailed profiling.

#### 6. Pipeline Step Function

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-core/src/pipeline/mod.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 617+ | `step()` trait method | Forward pass timing, cache operations |

**Recommended**: Instrument at pipeline level for model-agnostic metrics.

#### 7. Sampling/Token Generation

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-core/src/pipeline/sampling.rs`

| Location | Purpose | Attributes to Add |
|----------|---------|-------------------|
| Line 28-35 | `finish_or_add_toks_to_seq()` | Per-token events, stop reason |
| Line 69-200 | Streaming chunk handling | Chunk timing, tool call detection |

**Recommended**: Add events for token generation to capture fine-grained timing.

### Low-Priority Integration Points

#### 8. Model Loading

**File**: Various loader files in `/Users/cadence/Workspace/mistral.rs/mistralrs-core/src/pipeline/`

| Files | Purpose |
|-------|---------|
| `normal.rs`, `gguf.rs`, `vision.rs`, `embedding.rs` | Model initialization |

**Recommended**: Add spans for model loading time, memory allocation.

#### 9. KV Cache Operations

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-core/src/kv_cache/`

| Purpose | Attributes |
|---------|------------|
| Cache allocation | `llm.cache.type`, `llm.cache.size_bytes` |
| Prefix cache hits | `llm.token_count.prompt_details.cache_read` |

#### 10. Interactive Mode (CLI)

**File**: `/Users/cadence/Workspace/mistral.rs/mistralrs-server/src/interactive_mode.rs`

| Location | Purpose |
|----------|---------|
| Line 343-416 | TTFT measurement (already exists) |
| Line 666-739 | Completion mode TTFT |

**Note**: Already tracks TTFT locally; could export to telemetry.

---

## Recommended Instrumentation Strategy

### Phase 1: Complete Server-Layer Instrumentation

1. **Add missing `#[tracing::instrument]` to completions handler**
2. **Record token usage for all endpoints** (embeddings, reranking)
3. **Add streaming completion telemetry**:
   - Time-to-first-token (TTFT) as span event
   - Final token counts recorded before span closes

### Phase 2: Add Core Engine Telemetry

1. **Instrument engine step functions**:
   - Prompt processing time
   - Completion throughput (tokens/sec)
   - Batch size
2. **Add cache metrics**:
   - Prefix cache hit rate
   - KV cache utilization

### Phase 3: Add Advanced Observability

1. **Model loading telemetry**
2. **Memory allocation tracking**
3. **Distributed tracing for pipeline parallelism**

---

## Code Examples

### Example 1: Add Instrumentation to Completions Endpoint

```rust
// In mistralrs-server-core/src/completions.rs

use crate::telemetry::record_token_usage;

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses((status = 200, description = "Completions"))
)]
#[tracing::instrument(
    name = "completion",
    skip(state, oairequest),
    fields(
        otel.kind = "server",
        llm.model_name = %oairequest.model,
        openinference.span.kind = "LLM",
        llm.provider = "mistral.rs",
        gen_ai.request.model = %oairequest.model,
        gen_ai.provider.name = "mistral.rs",
        // Placeholders to be filled in later
        llm.token_count.prompt = tracing::field::Empty,
        llm.token_count.completion = tracing::field::Empty,
        llm.token_count.total = tracing::field::Empty,
        gen_ai.usage.input_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
    )
)]
pub async fn completions(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<CompletionRequest>,
) -> CompletionResponder {
    let (tx, mut rx) = create_response_channel(None);

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        CompletionResponder::Sse(create_streamer(rx, state, None, None))
    } else {
        let response = process_non_streaming_response(&mut rx, state).await;

        // Record token usage for non-streaming
        if let CompletionResponder::Json(ref json_resp) = response {
            record_token_usage(
                &tracing::Span::current(),
                json_resp.usage.prompt_tokens,
                json_resp.usage.completion_tokens,
            );
        }

        response
    }
}
```

### Example 2: Add TTFT Event for Streaming

```rust
// In mistralrs-server-core/src/streaming.rs or chat_completion.rs

use tracing::Span;

/// Record time-to-first-token as a span event
pub fn record_ttft_event(span: &Span, ttft_ms: f64) {
    // OpenInference defines TTFT as a span event
    span.in_scope(|| {
        tracing::info!(
            target: "openinference",
            name: "time_to_first_token",
            ttft_ms = ttft_ms,
            "First token generated"
        );
    });
}

// Usage in streaming handler:
impl futures::Stream for ChatCompletionStreamer {
    fn poll_next(...) -> Poll<Option<Self::Item>> {
        // ... existing code ...

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::Chunk(response) => {
                    // Record TTFT on first non-empty chunk
                    if self.first_chunk_time.is_none()
                        && response.choices.iter().any(|c| c.delta.content.is_some())
                    {
                        let ttft = self.start_time.elapsed();
                        self.first_chunk_time = Some(ttft);
                        record_ttft_event(&Span::current(), ttft.as_secs_f64() * 1000.0);
                    }
                    // ... rest of handling
                }
            }
        }
    }
}
```

### Example 3: Record Input/Output Messages

```rust
// In telemetry.rs - add helper for message recording

/// Record input messages on a span (OpenInference format)
pub fn record_input_messages(span: &Span, messages: &[IndexMap<String, MessageContent>]) {
    for (i, msg) in messages.iter().enumerate() {
        if let Some(Either::Left(role)) = msg.get("role") {
            span.record(
                &format!("llm.input_messages.{}.message.role", i),
                role.as_str(),
            );
        }
        if let Some(Either::Left(content)) = msg.get("content") {
            span.record(
                &format!("llm.input_messages.{}.message.content", i),
                content.as_str(),
            );
        }
    }
}

/// Record output message on a span
pub fn record_output_message(span: &Span, role: &str, content: &str, index: usize) {
    span.record(
        &format!("llm.output_messages.{}.message.role", index),
        role,
    );
    span.record(
        &format!("llm.output_messages.{}.message.content", index),
        content,
    );
}
```

### Example 4: Instrument Engine Forward Pass

```rust
// In mistralrs-core/src/engine/mod.rs

// Add a helper span for forward passes
fn create_forward_span(
    is_prompt: bool,
    batch_size: usize,
    total_tokens: usize,
) -> tracing::Span {
    if is_prompt {
        tracing::info_span!(
            "forward_prompt",
            otel.name = "forward prompt",
            batch_size = batch_size,
            total_tokens = total_tokens,
            "llm.timing.prompt_time_sec" = tracing::field::Empty,
            "llm.timing.prompt_throughput_tok_per_sec" = tracing::field::Empty,
        )
    } else {
        tracing::info_span!(
            "forward_completion",
            otel.name = "forward completion",
            batch_size = batch_size,
            "llm.timing.completion_time_sec" = tracing::field::Empty,
            "llm.timing.completion_throughput_tok_per_sec" = tracing::field::Empty,
        )
    }
}

// In the engine loop, wrap forward passes:
async fn run(self: Arc<Self>) {
    // ... existing code ...

    if !scheduled.prompt.is_empty() {
        let span = create_forward_span(
            true,
            scheduled.prompt.len(),
            scheduled.prompt.iter().map(|s| s.len()).sum(),
        );
        let _guard = span.enter();

        let prompt_exec_time = {
            // ... existing forward pass code ...
        };

        // Record timing on span
        span.record(
            "llm.timing.prompt_time_sec",
            prompt_exec_time.as_secs_f64(),
        );
        span.record(
            "llm.timing.prompt_throughput_tok_per_sec",
            total_tokens as f64 / prompt_exec_time.as_secs_f64(),
        );
    }
}
```

### Example 5: Add Sampling Parameters to Span

```rust
// Record inference parameters as invocation_parameters

use serde_json::json;

pub fn record_sampling_params(span: &Span, params: &SamplingParams) {
    let params_json = json!({
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "max_tokens": params.max_len,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "repetition_penalty": params.repetition_penalty,
    });

    span.record("llm.invocation_parameters", params_json.to_string().as_str());

    // Also record individual OTel GenAI attributes
    if let Some(temp) = params.temperature {
        span.record("gen_ai.request.temperature", temp as f64);
    }
    if let Some(max_tokens) = params.max_len {
        span.record("gen_ai.request.max_tokens", max_tokens as i64);
    }
}
```

---

## Environment Variables

The existing telemetry implementation respects these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | None | OTLP collector endpoint (e.g., `http://localhost:4317`) |
| `OTEL_SERVICE_NAME` | `mistral-rs` | Service name in traces |
| `OTEL_SERVICE_VERSION` | Package version | Service version |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Sampling ratio (0.0-1.0) |
| `OPENINFERENCE_HIDE_INPUTS` | `false` | Redact input content |
| `OPENINFERENCE_HIDE_OUTPUTS` | `false` | Redact output content |

---

## References

### OpenInference
- [OpenInference GitHub](https://github.com/Arize-ai/openinference)
- [Semantic Conventions Specification](https://arize-ai.github.io/openinference/spec/semantic_conventions.html)
- [LLM Spans Specification](https://github.com/Arize-ai/openinference/blob/main/spec/llm_spans.md)

### OpenTelemetry GenAI
- [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [GenAI Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [GenAI Metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)

### OpenLLMetry
- [OpenLLMetry GitHub](https://github.com/traceloop/openllmetry)
- [Traceloop Documentation](https://www.traceloop.com/docs/openllmetry/introduction)

### Rust Tracing
- [tracing crate documentation](https://docs.rs/tracing/latest/tracing/)
- [tracing-opentelemetry](https://docs.rs/tracing-opentelemetry/latest/tracing_opentelemetry/)

---

## Summary

mistral.rs has a solid foundation for OpenInference/OpenTelemetry integration with the existing `telemetry.rs` module. The key opportunities for enhancement are:

1. **Complete coverage**: Add `#[tracing::instrument]` to all endpoints
2. **Token usage recording**: Ensure all endpoints record token counts
3. **Streaming telemetry**: Add TTFT events and streaming-aware token counting
4. **Core engine instrumentation**: Add spans for forward passes and cache operations
5. **Sampling parameters**: Record inference parameters on spans

The modular approach using the `tracing` crate with OpenTelemetry export means telemetry can be incrementally enhanced without major architectural changes.
