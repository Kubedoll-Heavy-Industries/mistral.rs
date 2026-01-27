use std::{collections::HashMap, fmt::Display, sync::Arc};

use super::*;
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde_json::{json, Value};

/// A type which can be used as a chat request.
pub struct Chat {
    pub messages: Vec<IndexMap<String, MessageContent>>,
    pub attachments: ChatAttachments,
    pub controls: ChatControls,
    pub sampling_params: TokenSamplingParams,
    pub constraint: Constraint,
    pub logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
    pub return_logprobs: bool,
    pub return_raw_logits: bool,
    pub truncate_sequence: bool,
}

impl Chat {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            attachments: ChatAttachments::default(),
            controls: ChatControls::default(),
            sampling_params: TokenSamplingParams::deterministic(),
            constraint: Constraint::None,
            logits_processors: None,
            return_logprobs: false,
            return_raw_logits: false,
            truncate_sequence: false,
        }
    }

    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = truncate_sequence;
        self
    }

    pub fn into_inference_input(self, is_streaming: bool) -> InferenceInput {
        let Chat {
            messages,
            attachments,
            controls,
            sampling_params,
            constraint,
            logits_processors,
            return_logprobs,
            return_raw_logits,
            truncate_sequence,
        } = self;

        let thinking =
            ThinkingMode::from_options(controls.enable_thinking, controls.reasoning_effort);

        let mut op_attachments = Vec::new();
        for image in attachments.images {
            op_attachments.push(ChatAttachment::Image(image));
        }
        for audio in attachments.audios {
            op_attachments.push(ChatAttachment::Audio(audio));
        }

        let op = InferenceOperation::Chat {
            messages,
            attachments: op_attachments,
            thinking,
            sampling_params,
            return_logprobs,
            constraint,
            tools: controls.tools,
            tool_choice: controls.tool_choice,
            logits_processors,
            return_raw_logits,
            web_search_options: controls.web_search_options,
        };

        InferenceInput {
            op,
            exec: InferenceExec {
                is_streaming,
                truncate_sequence,
            },
            adapters: None,
        }
    }
}

impl Default for Chat {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatAttachments {
    pub images: Vec<DynamicImage>,
    pub audios: Vec<AudioInput>,
}

impl ChatAttachments {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatControls {
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub web_search_options: Option<WebSearchOptions>,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text (chat) messages.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
pub struct TextMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    enable_thinking: Option<bool>,
}

impl From<TextMessages> for Vec<IndexMap<String, MessageContent>> {
    fn from(value: TextMessages) -> Self {
        value.messages
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A chat message role.
pub enum TextMessageRole {
    User,
    Assistant,
    System,
    Tool,
    Custom(String),
}

impl Display for TextMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
            Self::Tool => write!(f, "tool"),
            Self::Custom(c) => write!(f, "{c}"),
        }
    }
}

impl Default for TextMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl TextMessages {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            enable_thinking: None,
        }
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }
}

impl From<TextMessages> for Chat {
    fn from(value: TextMessages) -> Self {
        Chat {
            messages: value.messages,
            attachments: ChatAttachments::default(),
            controls: ChatControls {
                enable_thinking: value.enable_thinking,
                ..ChatControls::default()
            },
            sampling_params: TokenSamplingParams::deterministic(),
            constraint: Constraint::None,
            logits_processors: None,
            return_logprobs: false,
            return_raw_logits: false,
            truncate_sequence: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Text (chat) messages with images and/or audios.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
pub struct VisionMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    audios: Vec<AudioInput>,
    enable_thinking: Option<bool>,
}

impl Default for VisionMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionMessages {
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            messages: Vec::new(),
            audios: Vec::new(),
            enable_thinking: None,
        }
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, images, vec![], model)
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, vec![], audios, model)
    }

    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        let config = model.config().unwrap();
        let prefixer = match &config.category {
            ModelCategory::Vision { prefixer } => prefixer,
            _ => {
                anyhow::bail!("`add_image_message` expects a vision model.")
            }
        };

        // Images
        let n_added_images = images.len();
        let image_indexes: Vec<usize> =
            (self.images.len()..self.images.len() + n_added_images).collect();
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let audio_indexes: Vec<usize> =
            (self.audios.len()..self.audios.len() + n_added_audios).collect();
        self.audios.extend(audios);

        if n_added_images > 0 || n_added_audios > 0 {
            // Build mixed content parts
            let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
            for _ in 0..n_added_images {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("image".to_string()),
                )]));
            }
            for _ in 0..n_added_audios {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("audio".to_string()),
                )]));
            }
            // Prefix the text with any media context
            let mut prefixed_text = text.to_string();
            if !image_indexes.is_empty() {
                prefixed_text = prefixer.prefix_image(image_indexes, &prefixed_text);
            }
            if !audio_indexes.is_empty() {
                prefixed_text = prefixer.prefix_audio(audio_indexes, &prefixed_text);
            }
            // Add the final text part
            content_vec.push(IndexMap::from([
                ("type".to_string(), Value::String("text".to_string())),
                ("text".to_string(), Value::String(prefixed_text)),
            ]));

            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Right(content_vec)),
            ]));
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(text.to_string())),
            ]));
        }
        Ok(self)
    }

    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self.images.clear();
        self.audios.clear();

        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }
}

impl From<VisionMessages> for Chat {
    fn from(value: VisionMessages) -> Self {
        Chat {
            messages: value.messages,
            attachments: ChatAttachments {
                images: value.images,
                audios: value.audios,
            },
            controls: ChatControls {
                enable_thinking: value.enable_thinking,
                ..ChatControls::default()
            },
            sampling_params: TokenSamplingParams::deterministic(),
            constraint: Constraint::None,
            logits_processors: None,
            return_logprobs: false,
            return_raw_logits: false,
            truncate_sequence: false,
        }
    }
}

#[derive(Clone)]
/// A way to add messages with finer control given.
///
/// This includes control over:
/// - Logits processors
/// - Constraints
/// - Logprobs
/// - Tools
/// - Sampling
/// - Enable thinking for models that support the configuration
pub struct RequestBuilder {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    audios: Vec<AudioInput>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    adapters: Vec<String>,
    return_logprobs: bool,
    constraint: Constraint,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    sampling_params: TokenSamplingParams,
    web_search_options: Option<WebSearchOptions>,
    enable_thinking: Option<bool>,
    truncate_sequence: bool,
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<TextMessages> for RequestBuilder {
    fn from(value: TextMessages) -> Self {
        Self {
            messages: value.messages,
            images: Vec::new(),
            audios: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: TokenSamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
            truncate_sequence: false,
        }
    }
}

impl From<VisionMessages> for RequestBuilder {
    fn from(value: VisionMessages) -> Self {
        Self {
            messages: value.messages,
            images: value.images,
            audios: value.audios,
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: TokenSamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
            truncate_sequence: false,
        }
    }
}

impl RequestBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            images: Vec::new(),
            audios: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: TokenSamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
            truncate_sequence: false,
        }
    }

    pub fn with_web_search_options(mut self, web_search_options: WebSearchOptions) -> Self {
        self.web_search_options = Some(web_search_options);
        self
    }

    /// Add a message to the request.
    ///
    /// For messages with tool calls, use [`Self::add_message_with_tool_call`].
    /// For messages with tool outputs, use [`Self::add_tool_message`].
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Add a message with the output of a tool call.
    pub fn add_tool_message(mut self, tool_content: impl ToString, tool_id: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            (
                "role".to_string(),
                Either::Left(TextMessageRole::Tool.to_string()),
            ),
            (
                "content".to_string(),
                Either::Left(tool_content.to_string()),
            ),
            (
                "tool_call_id".to_string(),
                Either::Left(tool_id.to_string()),
            ),
        ]));
        self
    }

    pub fn add_message_with_tool_call(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        tool_calls: Vec<ToolCallResponse>,
    ) -> Self {
        let tool_messages = tool_calls
            .iter()
            .map(|t| {
                IndexMap::from([
                    ("id".to_string(), Value::String(t.id.clone())),
                    ("type".to_string(), Value::String(t.tp.to_string())),
                    (
                        "function".to_string(),
                        json!({
                            "name": t.function.name,
                            "arguments": t.function.arguments,
                        }),
                    ),
                ])
            })
            .collect();
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
            ("function".to_string(), Either::Right(tool_messages)),
        ]));
        self
    }

    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, images, vec![], model)
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, vec![], audios, model)
    }

    /// By convention, all images are added before all audios.
    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        let config = model.config().unwrap();
        let prefixer = match &config.category {
            ModelCategory::Vision { prefixer } => prefixer,
            _ => {
                anyhow::bail!("`add_image_message` expects a vision model.")
            }
        };

        // Images
        let n_added_images = images.len();
        let image_indexes: Vec<usize> =
            (self.images.len()..self.images.len() + n_added_images).collect();
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let audio_indexes: Vec<usize> =
            (self.audios.len()..self.audios.len() + n_added_audios).collect();
        self.audios.extend(audios);

        if n_added_images > 0 || n_added_audios > 0 {
            // Build mixed content parts
            let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
            for _ in 0..n_added_images {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("image".to_string()),
                )]));
            }
            for _ in 0..n_added_audios {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("audio".to_string()),
                )]));
            }
            // Prefix the text with any media context
            let mut prefixed_text = text.to_string();
            if !image_indexes.is_empty() {
                prefixed_text = prefixer.prefix_image(image_indexes, &prefixed_text);
            }
            if !audio_indexes.is_empty() {
                prefixed_text = prefixer.prefix_audio(audio_indexes, &prefixed_text);
            }
            // Add the final text part
            content_vec.push(IndexMap::from([
                ("type".to_string(), Value::String("text".to_string())),
                ("text".to_string(), Value::String(prefixed_text)),
            ]));

            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Right(content_vec)),
            ]));
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(text.to_string())),
            ]));
        }
        Ok(self)
    }

    pub fn add_logits_processor(mut self, processor: Arc<dyn CustomLogitsProcessor>) -> Self {
        self.logits_processors.push(processor);
        self
    }

    pub fn set_adapters(mut self, adapters: Vec<String>) -> Self {
        self.adapters = adapters;
        self
    }

    /// The default tool choice is auto.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn return_logprobs(mut self, return_logprobs: bool) -> Self {
        self.return_logprobs = return_logprobs;
        self
    }

    pub fn set_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set the sampling parameters as given.
    pub fn set_sampling(mut self, params: TokenSamplingParams) -> Self {
        self.sampling_params = params;
        self
    }

    /// Set the sampling parameters for deterministic generation.
    /// This sets up the parameters so that there is:
    /// - No temperature, topk, topp, minp
    /// - No penalties, stop tokens, or logit bias
    /// - No maximum length
    pub fn set_deterministic_sampler(mut self) -> Self {
        self.sampling_params = TokenSamplingParams::deterministic();
        self
    }

    pub fn set_sampler_temperature(mut self, temperature: f64) -> Self {
        self.sampling_params.temperature = Some(temperature);
        self
    }

    pub fn set_sampler_topk(mut self, topk: usize) -> Self {
        self.sampling_params.top_k = Some(topk);
        self
    }

    pub fn set_sampler_topp(mut self, topp: f64) -> Self {
        self.sampling_params.top_p = Some(topp);
        self
    }

    pub fn set_sampler_minp(mut self, minp: f64) -> Self {
        self.sampling_params.min_p = Some(minp);
        self
    }

    pub fn set_sampler_topn_logprobs(mut self, top_n_logprobs: usize) -> Self {
        self.sampling_params.top_n_logprobs = top_n_logprobs;
        self
    }

    pub fn set_sampler_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.sampling_params.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn set_sampler_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.sampling_params.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn set_sampler_stop_toks(mut self, stop_toks: StopTokens) -> Self {
        self.sampling_params.stop_toks = Some(stop_toks);
        self
    }

    pub fn set_sampler_max_len(mut self, max_len: usize) -> Self {
        self.sampling_params.max_len = Some(max_len);
        self
    }

    pub fn set_sampler_logits_bias(mut self, logits_bias: HashMap<u32, f32>) -> Self {
        self.sampling_params.logits_bias = Some(logits_bias);
        self
    }

    pub fn set_sampler_n_choices(mut self, n_choices: usize) -> Self {
        self.sampling_params.n_choices = n_choices;
        self
    }

    pub fn set_sampler_dry_params(mut self, dry_params: DryTokenSamplingParams) -> Self {
        self.sampling_params.dry_params = Some(dry_params);
        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Truncate prompts that exceed the model's maximum context length.
    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = truncate_sequence;
        self
    }
}

impl From<RequestBuilder> for Chat {
    fn from(value: RequestBuilder) -> Self {
        let tools = if value.tools.is_empty() {
            None
        } else {
            Some(value.tools)
        };

        let tool_choice = if tools.is_some() {
            Some(value.tool_choice)
        } else {
            None
        };

        Chat {
            messages: value.messages,
            attachments: ChatAttachments {
                images: value.images,
                audios: value.audios,
            },
            controls: ChatControls {
                tools,
                tool_choice,
                web_search_options: value.web_search_options,
                enable_thinking: value.enable_thinking,
                reasoning_effort: None,
            },
            sampling_params: value.sampling_params,
            constraint: value.constraint,
            logits_processors: if value.logits_processors.is_empty() {
                None
            } else {
                Some(value.logits_processors)
            },
            return_logprobs: value.return_logprobs,
            return_raw_logits: false,
            truncate_sequence: value.truncate_sequence,
        }
    }
}

#[derive(Clone, Debug)]
/// An individual embedding input.
pub enum EmbeddingRequestInput {
    /// Raw text prompt that will be tokenized.
    Prompt(String),
    /// Pre-tokenized input.
    Tokens(Vec<u32>),
}

impl EmbeddingRequestInput {
    pub fn into_operation(self) -> mistralrs_core::InferenceOperation {
        match self {
            Self::Prompt(prompt) => mistralrs_core::InferenceOperation::Embedding { prompt },
            Self::Tokens(prompt) => mistralrs_core::InferenceOperation::EmbeddingTokens { prompt },
        }
    }
}

#[derive(Clone, Debug)]
/// A validated embedding request constructed via [`EmbeddingRequestBuilder`].
pub struct EmbeddingRequest {
    pub inputs: Vec<EmbeddingRequestInput>,
    pub truncate_sequence: bool,
}

impl EmbeddingRequest {
    /// Create a new builder for an embedding request.
    pub fn builder() -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::new()
    }
}

/// Builder for configuring embedding requests.
#[derive(Clone, Debug, Default)]
pub struct EmbeddingRequestBuilder {
    inputs: Vec<EmbeddingRequestInput>,
    truncate_sequence: bool,
}

impl EmbeddingRequestBuilder {
    /// Create an empty builder. You must add at least one input before using it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single text prompt.
    pub fn add_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inputs
            .push(EmbeddingRequestInput::Prompt(prompt.into()));
        self
    }

    /// Add multiple text prompts at once.
    pub fn add_prompts<I, S>(mut self, prompts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs.extend(
            prompts
                .into_iter()
                .map(|prompt| EmbeddingRequestInput::Prompt(prompt.into())),
        );
        self
    }

    /// Add a single pre-tokenized prompt.
    pub fn add_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
        self.inputs
            .push(EmbeddingRequestInput::Tokens(tokens.into()));
        self
    }

    /// Add multiple pre-tokenized prompts.
    pub fn add_tokens_batch<I>(mut self, batches: I) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        self.inputs
            .extend(batches.into_iter().map(EmbeddingRequestInput::Tokens));
        self
    }

    /// Control whether prompts longer than the model context are truncated.
    pub fn with_truncate_sequence(mut self, truncate: bool) -> Self {
        self.truncate_sequence = truncate;
        self
    }

    pub fn build(self) -> anyhow::Result<EmbeddingRequest> {
        if self.inputs.is_empty() {
            anyhow::bail!("Embedding request must contain at least one input.");
        }

        Ok(EmbeddingRequest {
            inputs: self.inputs,
            truncate_sequence: self.truncate_sequence,
        })
    }
}
