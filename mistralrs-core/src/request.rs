use either::Either;
use indexmap::IndexMap;
use mistralrs_audio::AudioInput;
use mistralrs_quant::IsqType;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::{
    pipeline::DiffusionGenerationParams, response::Response, sampler::TokenSamplingParams,
    tools::ToolChoice, CustomLogitsProcessor, Tool,
};
use std::{fmt::Debug, sync::Arc};
use tokio::sync::mpsc::Sender;

/// Model inference request.
///
/// Generic over input and response types:
/// - `I`: Input data (messages, tokens, etc.)
/// - `R`: Response type sent through the channel
///
/// The response channel is `Sender<R>`. For requests that need error handling,
/// use `R = anyhow::Result<T>`.
#[derive(Serialize, Deserialize)]
pub struct InferenceRequest<I, R> {
    pub id: uuid::Uuid,
    pub input: I,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<R>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

// Manual Clone impl - Sender<T> is Clone regardless of T, so we only need I: Clone
impl<I: Clone, R> Clone for InferenceRequest<I, R> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            input: self.input.clone(),
            response: self.response.clone(),
            model_id: self.model_id.clone(),
        }
    }
}

// =============================================================================
// Input types for inference requests
// =============================================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceExec {
    pub is_streaming: bool,
    pub truncate_sequence: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceInput {
    pub op: InferenceOperation,
    pub exec: InferenceExec,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ThinkingMode {
    Bool(bool),
    Effort(ReasoningEffort),
}

impl ThinkingMode {
    /// Construct ThinkingMode from optional enable flag and effort level.
    ///
    /// Priority: effort takes precedence over bool if both provided.
    pub fn from_options(
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Option<Self> {
        match (enable_thinking, reasoning_effort) {
            (_, Some(effort)) => Some(Self::Effort(effort)),
            (Some(b), None) => Some(Self::Bool(b)),
            (None, None) => None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ChatAttachment {
    #[serde(skip)]
    Image(image::DynamicImage),
    #[serde(skip)]
    Audio(AudioInput),
}

#[derive(Clone, Serialize, Deserialize)]
pub enum InferenceOperation {
    Chat {
        messages: Vec<IndexMap<String, MessageContent>>,
        attachments: Vec<ChatAttachment>,
        thinking: Option<ThinkingMode>,
        sampling_params: TokenSamplingParams,
        return_logprobs: bool,
        constraint: Constraint,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        #[serde(skip)]
        logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
        return_raw_logits: bool,
        web_search_options: Option<WebSearchOptions>,
    },
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: Option<usize>,
        sampling_params: TokenSamplingParams,
        return_logprobs: bool,
        constraint: Constraint,
        suffix: Option<String>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        #[serde(skip)]
        logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
        return_raw_logits: bool,
    },
    CompletionTokens {
        tokens: Vec<u32>,
        sampling_params: TokenSamplingParams,
        return_logprobs: bool,
        constraint: Constraint,
        suffix: Option<String>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        #[serde(skip)]
        logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
        return_raw_logits: bool,
    },
    ImageGeneration {
        prompt: String,
        format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
    },
    SpeechGeneration {
        prompt: String,
    },
    Embedding {
        prompt: String,
    },
    EmbeddingTokens {
        prompt: Vec<u32>,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
        truncate: bool,
    },
}

impl InferenceOperation {
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Chat { .. } => "chat",
            Self::Completion { .. } => "completion",
            Self::CompletionTokens { .. } => "completion_tokens",
            Self::ImageGeneration { .. } => "image_generation",
            Self::SpeechGeneration { .. } => "speech_generation",
            Self::Embedding { .. } => "embedding",
            Self::EmbeddingTokens { .. } => "embedding_tokens",
            Self::Rerank { .. } => "rerank",
        }
    }
}

/// Input for pipeline continuation requests (already tokenized).
///
/// Tokens accumulate in the sequence naturally (like single-node inference).
/// Position comes from seq.len(), not explicit tracking.
#[derive(Clone, Serialize, Deserialize)]
pub struct PipelineContinueInput {
    pub tokens: Vec<u32>,
    pub sampling_params: TokenSamplingParams,
    /// Initial sequence length (total prompt tokens).
    /// Prefill/decode boundary: seq.len() >= this value.
    pub initial_seq_len: usize,
}

/// Input for tokenization requests.
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenizeInput {
    pub text: Either<Vec<IndexMap<String, MessageContent>>, String>,
    pub tools: Option<Vec<Tool>>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Input for detokenization requests.
#[derive(Clone, Serialize, Deserialize)]
pub struct DetokenizeInput {
    pub tokens: Vec<u32>,
    pub skip_special_tokens: bool,
}

// =============================================================================
// Type aliases for concrete request types
// =============================================================================

/// Normal inference request (HTTP API → model inference → Response)
pub type NormalRequest = InferenceRequest<InferenceInput, Response>;

/// Pipeline continuation request (already tokenized → model inference → Response)
pub type PipelineRequest = InferenceRequest<PipelineContinueInput, Response>;

/// Tokenization request (text/messages → token IDs)
pub type TokenizeRequest = InferenceRequest<TokenizeInput, anyhow::Result<Vec<u32>>>;

/// Detokenization request (token IDs → text)
pub type DetokenizeRequest = InferenceRequest<DetokenizeInput, anyhow::Result<String>>;

// Backwards compatibility aliases
pub type TokenizationRequest = TokenizeRequest;
pub type DetokenizationRequest = DetokenizeRequest;
pub type PipelineContinueRequest = PipelineRequest;

pub type LlguidanceGrammar = llguidance::api::TopLevelGrammar;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Control the constraint with llguidance.
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(serde_json::Value),
    Llguidance(LlguidanceGrammar),
    None,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
/// Image generation response format
pub enum ImageGenerationResponseFormat {
    Url,
    B64Json,
}

pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Reasoning effort level for models that support it (e.g., GPT-OSS with Harmony format).
/// Controls the depth of reasoning/analysis in the model's response.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Minimal reasoning, faster responses
    Low,
    /// Balanced reasoning depth
    #[default]
    Medium,
    /// Deep reasoning, more thorough analysis
    High,
}

impl ReasoningEffort {
    /// Convert to string representation for chat template
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

fn default_responder<T>() -> Sender<T> {
    let (sender, _) = tokio::sync::mpsc::channel(1);
    sender
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[default]
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApproximateUserLocation {
    pub city: String,
    pub country: String,
    pub region: String,
    pub timezone: String,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum WebSearchUserLocation {
    #[serde(rename = "approximate")]
    Approximate {
        approximate: ApproximateUserLocation,
    },
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchOptions {
    pub search_context_size: Option<SearchContextSize>,
    pub user_location: Option<WebSearchUserLocation>,
    /// Override the description for the search tool.
    pub search_description: Option<String>,
    /// Override the description for the extraction tool.
    pub extract_description: Option<String>,
}


#[derive(Clone, Serialize, Deserialize)]
/// Discriminated union of all request types to the Engine.
/// Each variant wraps an InferenceRequest<I, R> with appropriate input/output types.
pub enum Request {
    /// Normal inference request (HTTP API)
    Normal(Box<NormalRequest>),
    /// Re-quantize model
    ReIsq(IsqType),
    /// Tokenization request
    Tokenize(TokenizeRequest),
    /// Detokenization request
    Detokenize(DetokenizeRequest),
    /// Pipeline continuation (distributed inference)
    PipelineContinue(PipelineRequest),
    /// Terminate the engine
    Terminate,
    /// Terminate all sequences on next step
    TerminateAllSeqsNextStep,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(boxed_req) => {
                let id = &boxed_req.id;
                let op_kind = boxed_req.input.op.kind_str();
                let is_streaming = boxed_req.input.exec.is_streaming;
                write!(
                    f,
                    "Request {id} {{ op: `{op_kind}`, is_streaming: {is_streaming}}}",
                )
            }
            Request::ReIsq(tp) => {
                write!(f, "Re ISQ Request {tp:?}",)
            }
            Request::Tokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.input.text)
            }
            Request::Detokenize(req) => {
                write!(f, "Detokenization Request {:?}", req.input.tokens)
            }
            Request::PipelineContinue(req) => {
                write!(f, "Pipeline Continue Request id={}", req.id)
            }
            Request::Terminate => write!(f, "Termination Request"),
            Request::TerminateAllSeqsNextStep => write!(f, "Terminate All Seqs Next Step"),
        }
    }
}
