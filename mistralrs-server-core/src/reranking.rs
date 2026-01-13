//! OpenAI-compatible reranking endpoint.
//!
//! Implements Cohere/Jina compatible `/v1/rerank` endpoint using cross-encoder models.
//!
//! Requires a reranker model to be loaded (e.g., `BAAI/bge-reranker-base`).
//! Returns an error if no reranker model is available.

use anyhow::{anyhow, Context, Error as AnyhowError};
use axum::{
    extract::{Json, State},
    http,
    response::IntoResponse,
};
use mistralrs_core::{
    MistralRs, NormalRequest, Request, Response,
};

use crate::{
    handler_core::{create_response_channel, send_request_with_model, ErrorToResponse, JsonError},
    openai::{RerankRequest, RerankResponse, RerankResult, RerankResultDocument, RerankUsage},
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::{sanitize_error_message, validate_model_name},
};

/// Represents different types of reranking responses.
pub enum RerankResponder {
    Json(RerankResponse),
    InternalError(AnyhowError),
    ValidationError(AnyhowError),
}

impl IntoResponse for RerankResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            RerankResponder::Json(s) => Json(s).into_response(),
            RerankResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.root_cause()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            RerankResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.root_cause()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/rerank",
    request_body = RerankRequest,
    responses((status = 200, description = "Reranking results", body = RerankResponse))
)]
#[tracing::instrument(
    name = "rerank",
    skip(state, request),
    fields(
        otel.kind = "server",
        llm.model_name = %request.model,
        openinference.span.kind = "RERANKER"
    )
)]
pub async fn rerank(
    State(state): ExtractedMistralRsState,
    Json(request): Json<RerankRequest>,
) -> RerankResponder {
    let repr = serde_json::to_string(&request).expect("Serialization of rerank request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&request.model, state.clone()) {
        return validation_error(e);
    }

    if request.documents.is_empty() {
        return validation_error(anyhow!("documents must contain at least one entry."));
    }

    let model_override = if request.model == "default" {
        None
    } else {
        Some(request.model.clone())
    };

    match cross_encoder_rerank(state.clone(), &request, model_override.as_deref()).await {
        Ok(response) => {
            MistralRs::maybe_log_response(state.clone(), &response);
            RerankResponder::Json(response)
        }
        Err(e) => {
            MistralRs::maybe_log_error(state.clone(), e.as_ref());
            internal_error(e)
        }
    }
}

/// Perform cross-encoder reranking using TEI backend
async fn cross_encoder_rerank(
    state: SharedMistralRsState,
    request: &RerankRequest,
    model_id: Option<&str>,
) -> anyhow::Result<RerankResponse> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let documents: Vec<String> = request
        .documents
        .iter()
        .map(|d| d.as_text().to_string())
        .collect();

    let rerank_request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        response: tx,
        model_id: model_id.map(|m| m.to_string()),
        input: mistralrs_core::InferenceInput {
            op: mistralrs_core::InferenceOperation::Rerank {
                query: request.query.clone(),
                documents: documents.clone(),
                truncate: true,
            },
            exec: mistralrs_core::InferenceExec {
                is_streaming: false,
                truncate_sequence: false,
            },
        },
    }));

    send_request_with_model(&state, rerank_request, model_id)
        .await
        .context("Failed to dispatch rerank request")?;

    // Process the response
    let response = rx.recv().await.ok_or_else(|| anyhow!("Channel closed"))?;

    match response {
        Response::Rerank {
            scores,
            prompt_tokens,
            total_tokens,
        } => {
            // Build results with scores
            let mut results: Vec<RerankResult> = scores
                .into_iter()
                .enumerate()
                .map(|(index, relevance_score)| {
                    let document = if request.return_documents {
                        Some(RerankResultDocument {
                            text: documents[index].clone(),
                        })
                    } else {
                        None
                    };
                    RerankResult {
                        index,
                        relevance_score,
                        document,
                    }
                })
                .collect();

            // Sort by relevance score descending
            results.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Apply top_n limit if specified
            if let Some(top_n) = request.top_n {
                results.truncate(top_n);
            }

            Ok(RerankResponse {
                model: request.model.clone(),
                results,
                usage: Some(RerankUsage {
                    prompt_tokens: saturating_to_u32(prompt_tokens),
                    total_tokens: saturating_to_u32(total_tokens),
                }),
            })
        }
        Response::ValidationError(e) => Err(anyhow!(e)),
        Response::InternalError(e) => Err(anyhow!(e)),
        _ => Err(anyhow!("Unexpected response type for rerank request")),
    }
}

fn validation_error<E>(err: E) -> RerankResponder
where
    E: Into<AnyhowError>,
{
    RerankResponder::ValidationError(err.into())
}

fn internal_error<E>(err: E) -> RerankResponder
where
    E: Into<AnyhowError>,
{
    RerankResponder::InternalError(err.into())
}

fn saturating_to_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}
