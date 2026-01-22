use crate::{
    pipeline::{ForwardInputsResult, NormalCache, RerankInputs},
    prefix_cacher::MatchingCache,
    request::{
        DetokenizeRequest, InferenceOperation, NormalRequest, PipelineContinueRequest, TokenizeRequest,
    },
    sequence::{SeqStepType, SequenceRecognizer, SequenceState},
    tools::{ToolCallingMatcher, ToolChoice},
    Constraint, ModelCategory, Response, TokenSamplingParams,
};
use candle_core::Tensor;
use either::Either;
use std::{
    ops::Deref,
    sync::{atomic::Ordering, Arc},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

use crate::{
    get_mut_arcmutex, handle_seq_error,
    request::Request,
    sampler::CustomLogitsProcessor,
    sequence::{Sequence, SequenceGroup},
    StopTokens,
};

use super::{search_request, Engine, TERMINATE_ALL_NEXT_STEP};

/// Send an error response and return from the current function.
/// Usage: `send_error!(response, Response::ValidationError("msg".into()))`
macro_rules! send_error {
    ($response:expr, $err:expr) => {{
        $response
            .send($err)
            .await
            .unwrap_or_else(|_| warn!("Receiver disconnected"));
        return;
    }};
}

/// Send an error response and return None from the current function.
/// Usage: `send_error_none!(response, Response::ValidationError("msg".into()))`
macro_rules! send_error_none {
    ($response:expr, $err:expr) => {{
        $response
            .send($err)
            .await
            .unwrap_or_else(|_| warn!("Receiver disconnected"));
        return None;
    }};
}

impl Engine {
    pub async fn handle_request(self: Arc<Self>, request: Request) {
        match request {
            Request::Normal(request) => {
                // Handle reranking separately - it uses TEI's predict() not generation
                if matches!(&request.input.op, InferenceOperation::Rerank { .. }) {
                    self.handle_rerank_request(*request).await;
                    return;
                }

                let is_chat = matches!(&request.input.op, InferenceOperation::Chat { .. });
                let has_tooling =
                    !self.tool_callbacks.is_empty() || !self.tool_callbacks_with_tools.is_empty();
                let has_search = match &request.input.op {
                    InferenceOperation::Chat {
                        web_search_options,
                        ..
                    } => web_search_options.is_some(),
                    _ => false,
                };

                if is_chat && (has_search || has_tooling) {
                    search_request::search_request(self.clone(), *request).await;
                } else {
                    self.add_request(*request).await;
                }
            }
            Request::ReIsq(level) => {
                if let Err(e) = get_mut_arcmutex!(self.pipeline).re_isq_model(level) {
                    warn!("ISQ requantization failed: {e:?}");
                }
            }
            Request::Tokenize(req) => self.tokenize_text(req).await,
            Request::Detokenize(req) => self.detokenize_text(req).await,
            Request::PipelineContinue(req) => self.handle_pipeline_continue(req).await,
            Request::Terminate => (),
            Request::TerminateAllSeqsNextStep => {
                TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst)
            }
        }
    }

    async fn handle_chat(
        &self,
        messages: Vec<indexmap::IndexMap<String, crate::request::MessageContent>>,
        thinking: Option<crate::request::ThinkingMode>,
        tools: Vec<crate::Tool>,
        response: &tokio::sync::mpsc::Sender<Response>,
    ) -> (Vec<u32>, String) {
        let pipeline = &*get_mut_arcmutex!(self.pipeline);
        let template = pipeline.get_processor().process(
            pipeline,
            messages,
            true,
            true,
            thinking,
            tools,
        );
        match template {
            Ok((toks, txt)) => (toks, txt),
            Err(e) => {
                response
                    .send(Response::InternalError(e.into()))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                (vec![], String::new())
            }
        }
    }

    async fn handle_text_generation(
        &self,
        text: String,
        response: &tokio::sync::mpsc::Sender<Response>,
    ) -> Option<(Vec<u32>, String)> {
        let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
            send_error_none!(response, Response::ValidationError(
                "Completion requests require the pipeline to have a tokenizer".into()
            ));
        };

        let prompt = tokenizer
            .encode_fast(text.clone(), true)
            .map_err(anyhow::Error::msg);
        let tokenized = match prompt {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                send_error_none!(response, Response::InternalError(e.into()));
            }
        };
        Some((tokenized, text))
    }

    pub(super) async fn add_request(&self, request: NormalRequest) {
        let is_chat = matches!(request.input.op, InferenceOperation::Chat { .. });
        let echo_prompt = matches!(
            request.input.op,
            InferenceOperation::Completion { echo_prompt, .. } if echo_prompt
        );

        let best_of = match &request.input.op {
            InferenceOperation::Completion { best_of, .. } => *best_of,
            _ => None,
        };
        let truncate_sequence = request.input.exec.truncate_sequence;
        if is_chat
            && !get_mut_arcmutex!(self.pipeline)
                .get_chat_template()
                .as_ref()
                .is_some_and(|ch_t| ch_t.has_chat_template())
        {
            send_error!(request.response, Response::ValidationError(
                "Received messages for a model which does not have a chat template. Either use a different model or pass a single string as the prompt".into()
            ));
        }

        // Verify the model's category matches the messages received.
        match (
            get_mut_arcmutex!(self.pipeline).category(),
            &request.input.op,
        ) {
            (
                ModelCategory::Text | ModelCategory::Vision { .. },
                InferenceOperation::Chat { .. }
                | InferenceOperation::Completion { .. }
                | InferenceOperation::CompletionTokens { .. },
            ) => (),
            (ModelCategory::Diffusion, InferenceOperation::ImageGeneration { .. }) => (),
            (ModelCategory::Speech, InferenceOperation::SpeechGeneration { .. }) => (),
            (
                ModelCategory::Embedding,
                InferenceOperation::Embedding { .. } | InferenceOperation::EmbeddingTokens { .. },
            ) => (),
            (ModelCategory::Rerank, InferenceOperation::Rerank { .. }) => (),
            _ => {
                send_error!(request.response, Response::ValidationError(
                    "Received a request incompatible for this model's category.".into()
                ));
            }
        }

        let (images, audios) = match &request.input.op {
            InferenceOperation::Chat { attachments, .. } => {
                let mut images = Vec::new();
                let mut audios = Vec::new();
                for attachment in attachments {
                    match attachment {
                        crate::request::ChatAttachment::Image(image) => images.push(image.clone()),
                        crate::request::ChatAttachment::Audio(audio) => audios.push(audio.clone()),
                    }
                }
                (
                    (!images.is_empty()).then_some(images),
                    (!audios.is_empty()).then_some(audios),
                )
            }
            _ => (None, None),
        };

        let tool_choice = match &request.input.op {
            InferenceOperation::Chat { tool_choice, .. }
            | InferenceOperation::Completion { tool_choice, .. }
            | InferenceOperation::CompletionTokens { tool_choice, .. } => {
                tool_choice.clone().unwrap_or(ToolChoice::Auto)
            }
            _ => ToolChoice::Auto,
        };

        let matcher = Arc::new(handle_seq_error!(
            ToolCallingMatcher::new(tool_choice,),
            request.response
        ));

        let image_generation_format = match &request.input.op {
            InferenceOperation::ImageGeneration { format, .. } => Some(*format),
            _ => None,
        };

        let seq_step_type = match &request.input.op {
            InferenceOperation::ImageGeneration { .. }
            | InferenceOperation::SpeechGeneration { .. }
            | InferenceOperation::Embedding { .. }
            | InferenceOperation::EmbeddingTokens { .. }
            | InferenceOperation::Rerank { .. } => SeqStepType::OneShot,
            _ => SeqStepType::PromptAndDecode,
        };

        let diffusion_params = match &request.input.op {
            InferenceOperation::ImageGeneration {
                generation_params, ..
            } => Some(generation_params.clone()),
            _ => None,
        };
        let mut added_seq = false;

        let (mut prompt_tokens, prompt_text) = match &request.input.op {
            InferenceOperation::Chat {
                messages,
                thinking,
                tools,
                ..
            } => {
                let tools = tools.clone().unwrap_or_default();
                self.handle_chat(
                    messages.clone(),
                    thinking.clone(),
                    tools,
                    &request.response,
                )
                .await
            }
            InferenceOperation::Completion { text, .. }
            | InferenceOperation::Embedding { prompt: text } => {
                let Some((prompt_tokens, prompt_text)) =
                    self.handle_text_generation(text.clone(), &request.response).await
                else {
                    return;
                };
                (prompt_tokens, prompt_text)
            }
            InferenceOperation::ImageGeneration { prompt, .. }
            | InferenceOperation::SpeechGeneration { prompt } => (vec![u32::MAX], prompt.clone()),
            InferenceOperation::CompletionTokens { tokens: it, .. }
            | InferenceOperation::EmbeddingTokens { prompt: it } => {
                let it = it.clone();
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    send_error!(request.response, Response::ValidationError(
                        "Completion requests w/ raw tokens require the pipeline to have a tokenizer".into()
                    ));
                };
                let prompt = tokenizer
                    .decode(&it, false)
                    .map_err(|e| anyhow::Error::msg(e.to_string()));
                (it, handle_seq_error!(prompt, request.response))
            }
            // Rerank is handled early in handle_request, should never reach here
            InferenceOperation::Rerank { .. } => {
                unreachable!("Rerank requests are handled by handle_rerank_request")
            }
        };
        if prompt_tokens.is_empty() {
            send_error!(request.response, Response::ValidationError(
                "Received an empty prompt.".into()
            ));
        }

        if matches!(
            get_mut_arcmutex!(self.pipeline).category(),
            ModelCategory::Text | ModelCategory::Vision { .. } | ModelCategory::Embedding
        ) && prompt_tokens.len() > get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len
        {
            // text/vision => truncate from start
            // embedding => truncate from end
            let category = get_mut_arcmutex!(self.pipeline).category();
            if !truncate_sequence {
                send_error!(request.response, Response::ValidationError(
                    format!("Prompt sequence length is greater than {}, perhaps consider using `truncate_sequence`?", get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len).into()
                ));
            } else if matches!(category, ModelCategory::Text | ModelCategory::Vision { .. }) {
                let prompt_len = prompt_tokens.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len;
                let currently_over = prompt_len - max_len;

                // Reserve space for generation tokens
                // If user specified max_len (generation length), reserve that many tokens (capped to max_len)
                // Otherwise, reserve just 1 token minimum to allow at least some generation
                let sampling_max = match &request.input.op {
                    InferenceOperation::Chat { sampling_params, .. }
                    | InferenceOperation::Completion { sampling_params, .. }
                    | InferenceOperation::CompletionTokens { sampling_params, .. } => sampling_params
                        .max_len
                        .unwrap_or(1)
                        .min(max_len),
                    _ => 1,
                };

                // Calculate how many prompt tokens to keep: max_len - sampling_max
                // This ensures we have room for generation
                let tokens_to_keep = max_len.saturating_sub(sampling_max);

                // Safely calculate slice start position - keep the end of the prompt
                let slice_start = prompt_len.saturating_sub(tokens_to_keep);

                prompt_tokens = prompt_tokens[slice_start..].to_vec();
                warn!("Prompt for request {} was {currently_over} tokens over the model maximum length. The first {slice_start} tokens were truncated to make space for generation.", request.id);
            } else {
                let prompt_len = prompt_tokens.len();
                let max_len = get_mut_arcmutex!(self.pipeline).get_metadata().max_seq_len;
                let currently_over = prompt_len - max_len;

                prompt_tokens = prompt_tokens[..max_len].to_vec();
                warn!("Prompt for request {} was {currently_over} tokens over the model maximum length. The last {currently_over} tokens were truncated to make space for generation.", request.id);
            }
        }

        let sampling_params = match &request.input.op {
            InferenceOperation::Chat { sampling_params, .. }
            | InferenceOperation::Completion { sampling_params, .. }
            | InferenceOperation::CompletionTokens { sampling_params, .. } => sampling_params,
            _ => &TokenSamplingParams::deterministic(),
        };

        let constraint = match &request.input.op {
            InferenceOperation::Chat { constraint, .. }
            | InferenceOperation::Completion { constraint, .. }
            | InferenceOperation::CompletionTokens { constraint, .. } => constraint,
            _ => &Constraint::None,
        };

        let suffix = match &request.input.op {
            InferenceOperation::Completion { suffix, .. }
            | InferenceOperation::CompletionTokens { suffix, .. } => suffix.clone(),
            _ => None,
        };

        let return_raw_logits = match &request.input.op {
            InferenceOperation::Chat {
                return_raw_logits, ..
            }
            | InferenceOperation::Completion {
                return_raw_logits, ..
            }
            | InferenceOperation::CompletionTokens {
                return_raw_logits, ..
            } => *return_raw_logits,
            _ => false,
        };

        let num_hidden_layers = get_mut_arcmutex!(self.pipeline)
            .get_metadata()
            .num_hidden_layers;

        let (stop_toks, stop_strings) = match sampling_params.stop_toks {
            None => (vec![], vec![]),
            Some(StopTokens::Ids(ref i)) => {
                let tok_env = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    pipeline.get_metadata().tok_env()
                };
                for id in i {
                    // We can't use ` ` (space) as a stop token because other tokens like ` moon` start with a space.
                    if let Some(tok_env) = tok_env.as_ref() {
                        let tok_trie = tok_env.tok_trie();
                        if tok_trie.has_extensions(tok_trie.token(*id)) {
                            send_error!(request.response, Response::ValidationError(
                                format!("Stop token {:?} is also a prefix of other tokens and cannot be used as a stop token.", tok_trie.token_str(*id)).into()
                            ));
                        }
                    }
                }

                (i.clone(), vec![])
            }
            Some(StopTokens::Seqs(ref s)) => {
                let mut stop_toks = Vec::new();
                let mut stop_strings: Vec<String> = Vec::new();

                let (tok_env, tokenizer) = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    let tok_env = pipeline.get_metadata().tok_env();
                    let tokenizer = pipeline.tokenizer();
                    (tok_env, tokenizer)
                };

                for stop_txt in s {
                    let Some(tokenizer) = &tokenizer else {
                        send_error!(request.response, Response::ValidationError(
                            "Completion requests require the pipeline to have a tokenizer".into()
                        ));
                    };
                    let encoded = tokenizer.encode_fast(stop_txt.to_string(), true);
                    let toks = handle_seq_error!(encoded, request.response)
                        .get_ids()
                        .to_vec();

                    if toks.len() == 1 {
                        if tok_env.as_ref().is_some_and(|tok_env| {
                            let tok_trie = tok_env.tok_trie();
                            tok_trie.has_extensions(tok_trie.token(toks[0]))
                        }) {
                            stop_strings.push(stop_txt.clone());
                        } else {
                            stop_toks.push(toks[0]);
                        }
                    } else {
                        stop_strings.push(stop_txt.clone());
                    }
                }

                (stop_toks, stop_strings)
            }
        };

        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            sampling_params.n_choices,
            request.input.exec.is_streaming,
            is_chat,
            best_of,
        )));

        let logits_processors: Vec<Arc<dyn CustomLogitsProcessor>> = match &request.input.op {
            InferenceOperation::Chat {
                logits_processors, ..
            }
            | InferenceOperation::Completion {
                logits_processors, ..
            }
            | InferenceOperation::CompletionTokens {
                logits_processors, ..
            } => logits_processors.clone().unwrap_or_default(),
            _ => vec![],
        };

        if sampling_params.n_choices == 0 {
            send_error!(request.response, Response::ValidationError(
                "n_choices must be greater than 0".into()
            ));
        }

        // Add sequences
        for response_index in 0..sampling_params.n_choices {
            let factory = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .llg_factory
                .clone();
            let recognizer = match Self::build_sequence_recognizer(&factory, constraint) {
                Ok(recognizer) => recognizer,
                Err(err) => {
                    send_error!(request.response, Response::ValidationError(
                        format!("Invalid grammar. {err}").into()
                    ));
                }
            };

            let block_size = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .cache_config
                .clone()
                .map(|conf| conf.block_size);

            let eos_toks = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .eos_tok
                .clone();

            let seq_preallocated_cache = if matches!(
                get_mut_arcmutex!(self.pipeline).category(),
                ModelCategory::Text | ModelCategory::Vision { .. }
            ) && get_mut_arcmutex!(self.pipeline)
                .do_preallocated_cache()
            {
                let metadata = get_mut_arcmutex!(self.pipeline).get_metadata();
                let model_metadata = metadata
                    .model_metadata
                    .as_ref()
                    .expect("If a model has a NormalCache it must have a model metadata");
                let n_tokens = prompt_tokens.len();
                let required_blocks = n_tokens.div_ceil(NormalCache::CACHE_GROW_SIZE);
                let max_seq_len = required_blocks * NormalCache::CACHE_GROW_SIZE;
                let k_shape = (
                    1usize,
                    model_metadata.num_kv_heads(),
                    max_seq_len,
                    model_metadata.k_head_dim(),
                );
                let v_shape = (
                    1usize,
                    model_metadata.num_kv_heads(),
                    max_seq_len,
                    model_metadata.v_head_dim(),
                );
                let dtype = get_mut_arcmutex!(self.pipeline)
                    .get_metadata()
                    .activation_dtype;

                let k_seq_cache = {
                    let k_seq_cache =
                        Tensor::zeros(k_shape, dtype, &get_mut_arcmutex!(self.pipeline).device());
                    match k_seq_cache {
                        Ok(x) => x,
                        Err(_) => {
                            send_error!(request.response, Response::InternalError(
                                "Failed to allocate preallocated KV cache.".to_string().into()
                            ));
                        }
                    }
                };
                let v_seq_cache = if k_shape == v_shape {
                    k_seq_cache.clone()
                } else {
                    let v_seq_cache =
                        Tensor::zeros(v_shape, dtype, &get_mut_arcmutex!(self.pipeline).device());
                    match v_seq_cache {
                        Ok(x) => x,
                        Err(_) => {
                            send_error!(request.response, Response::InternalError(
                                "Failed to allocate preallocated KV cache.".to_string().into()
                            ));
                        }
                    }
                };
                Some((k_seq_cache, v_seq_cache))
            } else {
                None
            };

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!");
            let mut seq = Sequence::new_waiting(
                prompt_tokens.clone(),
                prompt_text.clone(),
                *get_mut_arcmutex!(self.id).deref(),
                now.as_millis(),
                request.id, // UUID request_id for pipeline parallelism
                num_hidden_layers,
                request.response.clone(),
                sampling_params.clone(),
                logits_processors.clone(),
                stop_toks.clone(),
                stop_strings.clone(),
                sampling_params.max_len,
                match &request.input.op {
                    InferenceOperation::Chat { return_logprobs, .. }
                    | InferenceOperation::Completion { return_logprobs, .. }
                    | InferenceOperation::CompletionTokens { return_logprobs, .. } => {
                        *return_logprobs
                    }
                    _ => false,
                },
                get_mut_arcmutex!(self.pipeline).get_metadata().is_xlora,
                group.clone(),
                response_index,
                now.as_secs(),
                recognizer,
                suffix.clone(),
                if echo_prompt {
                    Some(prompt_text.clone())
                } else {
                    None
                },
                images.clone(),
                audios.clone(),
                block_size,
                Some(matcher.clone()),
                image_generation_format,
                seq_step_type,
                diffusion_params.clone(),
                seq_preallocated_cache,
                return_raw_logits,
                eos_toks,
                None, // pipeline_continue_op_id
                None, // logical_seq_len - normal requests use tokens.len()
            );

            // Only "track" a new sequence if it is a traditional one
            if matches!(seq_step_type, SeqStepType::PromptAndDecode) {
                self.logger.add_new_sequence();
            }

            // Enable Harmony mode if the chat template uses Harmony format
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if let Some(chat_template) = pipeline.get_chat_template() {
                    if chat_template.is_harmony_format() {
                        // Pre-warm the Harmony encoding if not already done.
                        // This must be done in a blocking context because openai-harmony
                        // uses reqwest::blocking which creates its own tokio runtime.
                        if !crate::harmony::is_harmony_encoding_ready() {
                            if let Err(e) = tokio::task::block_in_place(|| {
                                crate::harmony::prewarm_harmony_encoding();
                                Ok::<(), anyhow::Error>(())
                            }) {
                                warn!("Failed to initialize Harmony encoding: {e}");
                            }
                        }
                        if let Err(e) = seq.enable_harmony_mode() {
                            warn!("Failed to enable Harmony mode: {e}");
                        }
                    }
                }
            }

            // Allocate Mamba state pool slot for hybrid models
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                    let mut hybrid_cache = pipeline.cache().hybrid();
                    if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                        seq.set_mamba_state_idx(Some(slot_idx));
                    }
                }
            }

            // Run the inputs processor to update the prompt for multimodal models.
            if images.is_some() || audios.is_some() {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                let _ = pipeline.get_processor().inputs_processor().process_inputs(
                    pipeline.tokenizer(),
                    &mut [&mut seq],
                    true,
                    pipeline.get_metadata().is_xlora,
                    &pipeline.device(),
                    pipeline.get_metadata().no_kv_cache,
                    None,
                    false,
                    pipeline.get_input_processor_config(),
                    None,
                    pipeline.device_mapper(),
                );
            }

            let prefill_cache = handle_seq_error!(
                get_mut_arcmutex!(self.prefix_cacher).search_for_matching_cache(
                    seq.get_toks(),
                    seq.image_hashes(),
                    seq.audio_hashes(),
                ),
                request.response
            );

            seq = match prefill_cache.clone() {
                Some(MatchingCache::Normal {
                    normal,
                    images_to_keep,
                    audios_to_keep,
                    toks,
                    offset,
                }) => {
                    self.logger.add_prefix_cache_hit();

                    seq.keep_num_images(images_to_keep);
                    seq.keep_num_audios(audios_to_keep);
                    seq.prefill_v2_normal(normal, toks, offset)
                }
                Some(MatchingCache::Paged {
                    logical_blocks,
                    physical_blocks,
                    images_to_keep,
                    audios_to_keep,
                    toks,
                    offset,
                }) => {
                    self.logger.add_prefix_cache_hit();

                    seq.keep_num_images(images_to_keep);
                    seq.keep_num_audios(audios_to_keep);
                    seq.prefill_v2_paged(logical_blocks, physical_blocks, toks, offset)
                }
                None => seq,
            };

            // For pipeline parallelism: set pending tokens on hook before adding sequence
            // This applies to BOTH initial requests (first stage) AND continuation requests (later stages)
            let pipeline = get_mut_arcmutex!(self.pipeline);
            if let Some(hook_container) = pipeline.get_hook() {
                // Extract tokens from the sequence and set them on the hook
                // This allows the hook to propagate tokens with activations for sparse KV cache
                if let Some(hook) = hook_container.get() {
                    hook.set_pending_tokens(prompt_tokens.clone());
                    tracing::debug!(
                        token_count = prompt_tokens.len(),
                        "Tokens set on hook for pipeline parallelism"
                    );
                }
            }

            *get_mut_arcmutex!(self.id) += 1;
            get_mut_arcmutex!(self.scheduler).add_seq(seq);
            added_seq = true;
        }
        if added_seq {
            self.pending_notify.notify_one();
        }
    }

    async fn tokenize_text(&self, request: TokenizeRequest) {
        match request.input.text {
            Either::Left(messages) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tools = request.input.tools.unwrap_or_default();
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    request.input.add_generation_prompt,
                    request.input.add_special_tokens,
                    match (request.input.enable_thinking, request.input.reasoning_effort) {
                        (Some(b), None) => Some(crate::request::ThinkingMode::Bool(b)),
                        (None, Some(effort)) => {
                            Some(crate::request::ThinkingMode::Effort(effort))
                        }
                        (Some(_b), Some(effort)) => {
                            Some(crate::request::ThinkingMode::Effort(effort))
                        }
                        (None, None) => None,
                    },
                    tools,
                );
                let toks = match template {
                    Ok((toks, _)) => toks,
                    Err(e) => {
                        request
                            .response
                            .send(Err::<Vec<u32>, _>(e))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks))
                    .await
                    .expect("Sender disconnected unexpectedly!");
            }
            Either::Right(text) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tokenizer = pipeline.tokenizer();
                let tokenizer = match tokenizer {
                    Some(tokenizer) => tokenizer,
                    None => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(
                                "Pipeline does not include a toksnizer.",
                            )))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                let toks = tokenizer.encode_fast(text, request.input.add_special_tokens);
                let toks = match toks {
                    Ok(tokenizer) => tokenizer,
                    Err(e) => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(e)))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks.get_ids().to_vec()))
                    .await
                    .expect("Sender disconnected unexpectedly!");
            }
        };
    }

    async fn detokenize_text(&self, request: DetokenizeRequest) {
        let pipeline = &*get_mut_arcmutex!(self.pipeline);
        let tokenizer = pipeline.tokenizer();
        let tokenizer = match tokenizer {
            Some(tokenizer) => tokenizer,
            None => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(
                        "Pipeline does not include a tokenizer.",
                    )))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            }
        };
        let txt = tokenizer.decode(&request.input.tokens, request.input.skip_special_tokens);
        let txt = match txt {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(e)))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            }
        };
        request
            .response
            .send(Ok(txt))
            .await
            .expect("Sender disconnected unexpectedly!");
    }

    /// Handle a reranking request using the Pipeline interface.
    ///
    /// Reranking is handled separately from the standard generation pipeline because:
    /// 1. It uses classification (predict) rather than generation
    /// 2. Input is (query, document) pairs, not single prompts
    /// 3. Output is relevance scores, not generated tokens
    async fn handle_rerank_request(&self, request: NormalRequest) {
        let (query, documents, truncate) = match &request.input.op {
            InferenceOperation::Rerank {
                query,
                documents,
                truncate,
            } => (query.clone(), documents.clone(), *truncate),
            _ => {
                send_error!(request.response, Response::InternalError(
                    "handle_rerank_request called with non-Rerank message".into()
                ));
            }
        };

        // Validate model category
        let mut pipeline = get_mut_arcmutex!(self.pipeline);
        if !matches!(pipeline.category(), ModelCategory::Rerank) {
            send_error!(request.response, Response::ValidationError(
                "Loaded model is not a reranker. Use ModelSelected::Rerank to load a cross-encoder model.".into()
            ));
        }

        // Perform reranking using Pipeline::forward_inputs()
        let inputs = Box::new(RerankInputs {
            query,
            documents,
            truncate,
        });

        match pipeline.forward_inputs(inputs, false) {
            Ok(ForwardInputsResult::Rerank {
                scores,
                prompt_tokens,
                total_tokens,
            }) => {
                // Convert tensor to Vec<f32>
                let scores_vec = match scores.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(e) => {
                        send_error!(request.response, Response::InternalError(
                            format!("Failed to convert scores: {e}").into()
                        ));
                    }
                };

                request
                    .response
                    .send(Response::Rerank {
                        scores: scores_vec,
                        prompt_tokens,
                        total_tokens,
                    })
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
            }
            Ok(_) => {
                send_error!(request.response, Response::InternalError(
                    "RerankPipeline returned unexpected result type".into()
                ));
            }
            Err(e) => {
                send_error!(request.response, Response::InternalError(
                    format!("Reranking failed: {e}").into()
                ));
            }
        }
    }

    /// Handle pipeline continuation request for non-first pipeline stages.
    ///
    /// Sequences live in the scheduler for the entire request duration. This function:
    /// 1. First activation: Creates sequence and adds to scheduler
    /// 2. Subsequent activations: Updates existing sequence with new tokens
    ///
    /// The actual forward pass is done by Engine::run() - the hook blocks until
    /// activation data is available, which we set here.
    async fn handle_pipeline_continue(&self, request: PipelineContinueRequest) {
        let request_id = request.id;
        let tokens = request.input.tokens.clone();
        let initial_seq_len = request.input.initial_seq_len;

        // Set pending tokens on hook for activation injection
        {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            if let Some(hook_container) = pipeline.get_hook() {
                if let Some(hook) = hook_container.get() {
                    hook.set_pending_tokens(tokens.clone());
                }
            }
        }

        // Check if we have an existing sequence for this request in the scheduler
        let is_first_activation = {
            let scheduler = get_mut_arcmutex!(self.scheduler);
            !scheduler.has_sequence(request_id)
        };

        // Check if this is first stage (HEAD) - needs external logits from downstream.
        let is_first_stage = {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            pipeline
                .get_hook()
                .is_some_and(|h| h.needs_external_logits())
        };

        tracing::info!(
            %request_id,
            is_first_activation,
            is_first_stage,
            token_count = tokens.len(),
            initial_seq_len,
            "PP activation: tokens accumulate naturally"
        );

        if is_first_activation {
            // First activation: Create new sequence and add to scheduler
            let metadata = get_mut_arcmutex!(self.pipeline).get_metadata().clone();
            let num_hidden_layers = metadata.num_hidden_layers;
            let eos_toks = metadata.eos_tok.clone();

            // Get max_len from sampling params, falling back to model's max_seq_len
            let max_len = request
                .input
                .sampling_params
                .max_len
                .or(Some(metadata.max_seq_len));

            let sampling_params = request.input.sampling_params.clone();

            let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(1, false, false, None)));

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!");

            let block_size = {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                scheduler.block_size()
            };

            let mut seq = Sequence::new_waiting(
                tokens.clone(),
                String::new(),
                *get_mut_arcmutex!(self.id).deref(),
                now.as_millis(),
                request_id,
                num_hidden_layers,
                request.response.clone(),
                sampling_params,
                vec![], // logits_processors
                vec![], // stop_tokens
                vec![], // stop_strings
                max_len,
                false, // return_logprobs
                false, // is_xlora
                group,
                0,
                now.as_secs(),
                SequenceRecognizer::None,
                None, // suffix
                None, // prefix
                None, // input_images
                None, // input_audios
                block_size,
                None, // tools
                None, // image_gen_response_format
                SeqStepType::PromptAndDecode,
                None, // diffusion_params
                None, // seq_preallocated_cache
                true, // return_raw_logits
                eos_toks,
                None, // pipeline_continue_op_id
                Some(initial_seq_len), // logical_seq_len - PP continuation uses initial_seq_len
            );

            // token_offset starts at 0 for prefill. After prefill completes, the engine's
            // state transition logic will update it (see mod.rs lines 513-532).
            // Note: PP sequences (return_raw_logits=true) skip auto state transition,
            // so we rely on subsequent PipelineContinue to transition to RunningCompletion.
            seq.set_state(SequenceState::RunningPrompt);

            if block_size.is_some() {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                if let Some(block_engine) = scheduler.block_engine() {
                    get_mut_arcmutex!(block_engine).allocate(&mut seq);
                }
            }

            *get_mut_arcmutex!(self.id) += 1;

            // Add sequence to scheduler - Engine::run() will process it
            get_mut_arcmutex!(self.scheduler).add_seq(seq);

            // Wake up engine to process the new sequence
            self.pending_notify.notify_one();
        } else {
            // Subsequent activation: Append tokens to existing sequence
            let mut scheduler = get_mut_arcmutex!(self.scheduler);
            let block_engine = scheduler.block_engine();

            if let Some(seq) = scheduler.get_sequence_mut(request_id) {
                if !is_first_stage {
                    // TAIL stages: append tokens (like single-node inference)
                    seq.append_tokens(&tokens);

                    // Allocate blocks for new tokens
                    if let Some(block_engine) = &block_engine {
                        let mut engine = get_mut_arcmutex!(block_engine);
                        engine.allocate(seq);
                    }
                }

                // Update responder for response routing
                seq.set_responder(request.response.clone());

                // Determine prefill vs decode from token count, not accumulated length.
                // - Prefill chunk: tokens.len() > 1 (multiple tokens in activation)
                // - Decode: tokens.len() == 1 (single token per step)
                // This correctly handles TAIL's first activation (prefill) even when
                // seq.len() == initial_seq_len, which would incorrectly trigger decode.
                let is_decode = tokens.len() == 1 && seq.len() >= initial_seq_len;
                if is_decode {
                    // Transitioning to decode: set token_offset once
                    if matches!(seq.getstate(), SequenceState::RunningPrompt) {
                        seq.set_token_offset(initial_seq_len);
                    }
                    seq.set_state(SequenceState::RunningCompletion);
                } else {
                    // Prefill chunk: token_offset stays at 0, RoPE positions from make_prompt_chunk
                    seq.set_state(SequenceState::RunningPrompt);
                }
            } else {
                tracing::error!(%request_id, "Sequence not found in scheduler for update");
                let _ = request
                    .response
                    .send(Response::InternalError(
                        "Pipeline sequence not found".to_string().into(),
                    ))
                    .await;
            }
        }

        // Engine::run() will do the actual forward pass - the hook blocks until
        // activation data is available (which we just set above)
    }

}
