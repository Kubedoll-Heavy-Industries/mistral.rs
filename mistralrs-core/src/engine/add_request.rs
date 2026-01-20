use crate::{
    pipeline::{CacheBackendMetadata, CacheInstruction, ForwardInputsResult, NormalCache, RerankInputs, text_models_inputs_processor::PagedAttentionMeta},
    prefix_cacher::MatchingCache,
    request::{
        DetokenizeRequest, InferenceOperation, NormalRequest, PipelineContinueRequest, TokenizeRequest,
    },
    sequence::{SeqStepType, SequenceRecognizer, SequenceState},
    tools::{ToolCallingMatcher, ToolChoice},
    Constraint, ModelCategory, Response, SamplingParams,
};
use candle_core::Tensor;
use either::Either;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::{
    collections::HashMap,
    ops::Deref,
    sync::{atomic::Ordering, Arc},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

use crate::{
    get_mut_arcmutex, handle_seq_error,
    paged_attention::BlockEngineSequence,
    request::Request,
    sampler::Sampler,
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
            Request::PipelineCleanup { request_id } => {
                self.handle_pipeline_cleanup(request_id).await
            }
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
            _ => &SamplingParams::deterministic(),
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

        let topk = sampling_params.top_k.map(|x| x as i64).unwrap_or(-1);
        let topp = sampling_params.top_p.unwrap_or(1.0);
        let minp = sampling_params.min_p.unwrap_or(0.0);
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

        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let sampler = Sampler::new(
            Some(sampling_params.temperature.unwrap_or(1.0)),
            sampling_params.top_n_logprobs,
            tokenizer,
            sampling_params.frequency_penalty,
            sampling_params.presence_penalty,
            sampling_params.repetition_penalty,
            sampling_params.dry_params.clone(),
            topk,
            topp,
            minp,
            match &request.input.op {
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
            },
        );
        let sampler = handle_seq_error!(sampler, request.response);

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
                sampler.clone(),
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
            );

            // Pipeline continuation: configure the Sequence for PP non-first stages.
            // Use sequence_position to set token_offset for correct RoPE position calculation.
            if let Ok(mut meta) = self.pipeline_continue_meta.lock() {
                if let Some(pipeline_meta) = meta.remove(&request.id) {
                    // Set the total prompt length for prefill/decode boundary detection
                    seq.set_prompt_len(pipeline_meta.initial_seq_len);
                    // Set token_offset for correct position calculation in RoPE
                    // During decode, sequence_position equals total tokens processed,
                    // so token_offset ensures RoPE uses the correct position.
                    seq.set_token_offset(pipeline_meta.sequence_position);

                    // CRITICAL: Set prefill_chunk_size to enable KV cache preservation.
                    // The engine uses prefill_chunk_size.is_some() to identify sequences
                    // that need their KV cache cached/injected across decode steps.
                    // Value doesn't affect chunking behavior here - just enables the flag.
                    seq.set_prefill_chunk_size(Some(prompt_tokens.len()));

                    tracing::debug!(
                        %request.id,
                        initial_seq_len = pipeline_meta.initial_seq_len,
                        sequence_position = pipeline_meta.sequence_position,
                        token_count = prompt_tokens.len(),
                        "Configured Sequence for pipeline continuation"
                    );
                }
            }

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

            // For continuation requests, inject cached KV if available.
            // This preserves KV cache across decode steps in pipeline parallelism.
            // Presence in pipeline_kv_cache indicates this is a continuation request.
            {
                let op_id = request.id;
                if let Ok(cache_map) = self.pipeline_kv_cache.lock() {
                    if let Some(cached_kv) = cache_map.get(&op_id) {
                        tracing::debug!(
                            %op_id,
                            "Checking for cached KV to inject"
                        );
                        // Calculate cached length first
                        let cached_len = cached_kv.get(0)
                            .and_then(|kv| kv.as_ref())
                            .and_then(|kv| kv.k().ok().flatten())
                            .map(|k| k.dims()[2])
                            .unwrap_or(0);

                        // Inject cached KV into new sequence before adding to scheduler
                        let injected_cache_len = {
                            let new_kv = seq.normal_cache();
                            for (i, kv) in cached_kv.iter().enumerate() {
                                if i < new_kv.len() {
                                    new_kv[i] = kv.clone();
                                }
                            }

                            new_kv.get(0)
                                .and_then(|kv| kv.as_ref())
                                .and_then(|kv| kv.k().ok().flatten())
                                .map(|k| k.dims()[2])
                                .unwrap_or(0)
                        };

                        tracing::debug!(
                            %op_id,
                            cache_layers = cached_kv.iter().filter(|kv| kv.is_some()).count(),
                            seq_tokens = seq.get_toks().len(),
                            cached_len,
                            injected_cache_len,
                            "Cache injected for continuation"
                        );
                    }
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
    /// This method directly manages persistent sequences for pipeline parallelism,
    /// bypassing the scheduler. Sequences are stored in `pipeline_sequences` and
    /// reused across multiple forward passes (one per activation).
    ///
    /// Flow:
    /// 1. First activation: Create sequence, store in `pipeline_sequences`, run forward
    /// 2. Subsequent activations: Reuse sequence from `pipeline_sequences`, update tokens, run forward
    /// 3. Cleanup signal: Remove from `pipeline_sequences`
    async fn handle_pipeline_continue(&self, request: PipelineContinueRequest) {
        let request_id = request.id;
        let tokens = request.input.tokens.clone();
        let sequence_position = request.input.sequence_position;
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

        // Check if we have an existing sequence for this request
        let is_first_activation = {
            let seqs = self.pipeline_sequences.lock().unwrap();
            !seqs.contains_key(&request_id)
        };

        // Check if this is first stage (HEAD) - needs external logits from downstream.
        // First stage runs sampling locally, so tokens accumulate via add_token().
        // Non-first stages receive tokens from upstream and must track them explicitly.
        let is_first_stage = {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            pipeline
                .get_hook()
                .is_some_and(|h| h.needs_external_logits())
        };

        if is_first_activation {
            // First activation: Create new sequence with full setup
            let num_hidden_layers = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .num_hidden_layers;

            let eos_toks = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .eos_tok
                .clone();

            let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();
            let sampler = Sampler::new(
                Some(request.input.sampling_params.temperature.unwrap_or(1.0)),
                request.input.sampling_params.top_n_logprobs,
                tokenizer,
                request.input.sampling_params.frequency_penalty,
                request.input.sampling_params.presence_penalty,
                request.input.sampling_params.repetition_penalty,
                request.input.sampling_params.dry_params.clone(),
                request.input.sampling_params.top_k.map(|x| x as i64).unwrap_or(-1),
                request.input.sampling_params.top_p.unwrap_or(1.0),
                request.input.sampling_params.min_p.unwrap_or(0.0),
                vec![],
            );

            let sampler = match sampler {
                Ok(s) => s,
                Err(e) => {
                    let _ = request.response.send(Response::InternalError(e.into())).await;
                    return;
                }
            };

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
                sampler,
                vec![],
                vec![],
                None,
                false,
                false,
                group,
                0,
                now.as_secs(),
                SequenceRecognizer::None,
                None,
                None,
                None,
                None,
                block_size,
                None,
                None,
                SeqStepType::PromptAndDecode,
                None,
                None,
                true,
                eos_toks,
                None, // pipeline_continue_op_id
            );

            seq.set_prompt_len(initial_seq_len);
            seq.set_token_offset(sequence_position);
            seq.set_state(SequenceState::RunningPrompt);

            if block_size.is_some() {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                if let Some(block_engine) = scheduler.block_engine() {
                    get_mut_arcmutex!(block_engine).allocate(&mut seq);
                }
            }

            *get_mut_arcmutex!(self.id) += 1;

            {
                let mut seqs = self.pipeline_sequences.lock().unwrap();
                seqs.insert(request_id, seq);
            }
        } else {
            // Subsequent activation: Handle prefill chunks vs decode steps differently.
            let is_prompt_chunk = sequence_position < initial_seq_len;
            let mut seqs = self.pipeline_sequences.lock().unwrap();
            if let Some(seq) = seqs.get_mut(&request_id) {
                if is_first_stage {
                    // HEAD (Stage 0): Don't modify tokens or offset.
                    // Tokens accumulate naturally via add_token() during sampling.
                } else {
                    // TAIL stages: receive tokens from upstream
                    seq.receive_tokens(&tokens, is_prompt_chunk);
                    if is_prompt_chunk {
                        seq.set_token_offset(sequence_position);
                    }
                }
                // Update responder so response goes to THIS request's channel
                seq.set_responder(request.response.clone());
                // Prefill chunks stay in prompt state; decode transitions to completion
                if is_prompt_chunk {
                    seq.set_state(SequenceState::RunningPrompt);
                } else {
                    seq.set_state(SequenceState::RunningCompletion);
                }
            }
        }

        // Run forward pass directly, bypassing scheduler
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));

        // Phase 1: Extract sequence from map to release lock during await
        // This prevents deadlock when multiple pipeline requests run concurrently
        let mut seq = {
            let mut seqs = self.pipeline_sequences.lock().unwrap();
            match seqs.remove(&request_id) {
                Some(s) => s,
                None => {
                    tracing::error!(%request_id, "Sequence not found in pipeline_sequences");
                    let _ = request
                        .response
                        .send(Response::InternalError(
                            "Pipeline sequence not found".to_string().into(),
                        ))
                        .await;
                    return;
                }
            }
        };

        // Phase 2: Check first forward status (brief lock)
        let is_first_forward = {
            let done = self.pipeline_first_forward_done.lock().unwrap();
            !done.contains(&(request_id, 0))
        };

        // Determine prompt vs decode based on absolute sequence position.
        // For pipeline continuation, later prompt chunks must still be treated as prompt
        // (i.e., build KV) even though they are not the first forward.
        let is_prompt_step = sequence_position < initial_seq_len;

        // Construct backend_metadata based on whether PagedAttention is enabled
        let backend_metadata = {
            let scheduler = get_mut_arcmutex!(self.scheduler);
            if let (Some(block_engine), Some(block_size)) = (scheduler.block_engine(), scheduler.block_size()) {
                // PagedAttention is enabled - allocate/manage blocks and use PagedAttention metadata
                let sliding_window = get_mut_arcmutex!(self.pipeline).get_metadata().sliding_window;

                // For PagedAttention, block allocation happens during input processing
                // The block_engine is passed via PagedAttentionMeta and handles allocation internally
                let metadata = PagedAttentionMeta {
                    block_size,
                    sliding_window,
                    block_engine: block_engine.clone(),
                };

                // Handle block allocation for TAIL stages (non-first stage).
                // - Prefill: free + allocate (KV cache rebuilt from chunk tokens)
                // - Decode: preserve existing blocks + grow (KV cache must be preserved)
                if !is_first_activation && !is_first_stage {
                    let mut engine = get_mut_arcmutex!(block_engine);
                    if is_prompt_step {
                        // TAIL prefill: tokens were replaced, rebuild blocks from scratch
                        engine.free_sequence(*seq.id());
                        engine.allocate(&mut seq);
                    } else {
                        // TAIL decode: preserve existing blocks, grow if needed
                        // Take current block table and set as prefill blocks so allocate() preserves them
                        if let Some(existing_blocks) = engine.block_tables.remove(seq.id()) {
                            seq.set_physical_blocks_prefill(existing_blocks);
                            engine.allocate(&mut seq);
                        }
                    }
                }

                CacheBackendMetadata::PagedAttention {
                    metadata,
                    blocks_to_copy: HashMap::new(),
                }
            } else {
                // DefaultScheduler - use cache instructions
                let pre_op = if is_first_forward {
                    CacheInstruction::Reset {
                        load_preallocated_cache: false,
                        reset_non_granular: false,
                    }
                } else {
                    CacheInstruction::In
                };
                let post_op = CacheInstruction::Out;
                CacheBackendMetadata::DefaultInstructions { pre_op, post_op }
            }
        };

        // Phase 3: Execute pipeline operation (is_first_stage already computed above)
        // Pipeline lock is held during operation but pipeline_sequences is free
        let mut seq_refs: Vec<&mut Sequence> = vec![&mut seq];

        if is_first_stage {
            // Stage 0: Run full step (forward + wait for logits + sample + send response)
            let step_result = {
                let mut pipeline = get_mut_arcmutex!(self.pipeline);
                let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
                pipeline
                    .step(
                        &mut seq_refs,
                        is_prompt_step,
                        false, // Don't return raw logits - we want sampling
                        &mut *prefix_cacher,
                        self.disable_eos_stop,
                        rng,
                        backend_metadata,
                    )
                    .await
            };

            // Mark first forward as done
            {
                let mut done = self.pipeline_first_forward_done.lock().unwrap();
                done.insert((request_id, 0));
            }

            if let Err(e) = step_result {
                tracing::error!(%request_id, error = %e, "Pipeline step failed");
                let _ = request
                    .response
                    .send(Response::InternalError(
                        format!("Pipeline step failed: {e}").into(),
                    ))
                    .await;
                // Don't re-insert on error - cleanup will handle it
                return;
            }

            tracing::debug!(%request_id, "Pipeline step completed");
        } else {
            // Intermediate/last stage: Run forward pass and send logits to worker
            let result = {
                let mut pipeline = get_mut_arcmutex!(self.pipeline);
                pipeline.forward_pass(&mut seq_refs, is_prompt_step, backend_metadata)
            };

            // Mark first forward as done
            {
                let mut done = self.pipeline_first_forward_done.lock().unwrap();
                done.insert((request_id, 0));
            }

            match result {
                Ok(logits) => {
                    // Move logits to CPU before sending through channel.
                    // This must happen on the model thread which has the CUDA/Metal context.
                    // The receiving tokio task won't have GPU context.
                    let cpu_logits = match logits.to_device(&candle_core::Device::Cpu) {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::error!(%request_id, error = %e, "Failed to move logits to CPU");
                            let _ = request
                                .response
                                .send(Response::InternalError(
                                    format!("Failed to move logits to CPU: {e}").into(),
                                ))
                                .await;
                            return;
                        }
                    };
                    // Send CPU logits tensor in Response::Raw
                    let _ = request
                        .response
                        .send(Response::Raw {
                            logits_chunks: vec![cpu_logits],
                            tokens: vec![],
                        })
                        .await;
                    tracing::debug!(
                        %request_id,
                        "Pipeline forward pass completed, sent logits to worker"
                    );
                }
                Err(e) => {
                    tracing::error!(%request_id, error = %e, "Pipeline forward pass failed");
                    let _ = request
                        .response
                        .send(Response::InternalError(
                            format!("Pipeline forward failed: {e}").into(),
                        ))
                        .await;
                    // Don't re-insert on error - cleanup will handle it
                    return;
                }
            }
        }

        // Phase 5: Re-insert sequence for future use on success
        self.pipeline_sequences
            .lock()
            .unwrap()
            .insert(request_id, seq);
    }

    /// Handle cleanup signal for a pipeline request.
    /// Called when the stream closes (request completed or aborted).
    /// Removes all cached state associated with the request.
    async fn handle_pipeline_cleanup(&self, request_id: uuid::Uuid) {
        tracing::debug!(%request_id, "Pipeline cleanup requested");

        // Remove from pipeline_sequences and free associated PagedAttention blocks
        if let Ok(mut seqs) = self.pipeline_sequences.lock() {
            if let Some(seq) = seqs.remove(&request_id) {
                tracing::debug!(%request_id, seq_id = *seq.id(), "Removed sequence from pipeline_sequences");
                // Free PagedAttention blocks allocated for this sequence
                let scheduler = get_mut_arcmutex!(self.scheduler);
                if let Some(block_engine) = scheduler.block_engine() {
                    get_mut_arcmutex!(block_engine).free_sequence(*seq.id());
                    tracing::debug!(%request_id, seq_id = *seq.id(), "Freed PagedAttention blocks");
                }
            }
        }

        // Remove from pipeline_kv_cache (legacy KV cache storage)
        if let Ok(mut cache) = self.pipeline_kv_cache.lock() {
            if cache.remove(&request_id).is_some() {
                tracing::debug!(%request_id, "Removed KV cache from pipeline_kv_cache");
            }
        }

        // Remove from pipeline_continue_meta
        if let Ok(mut meta) = self.pipeline_continue_meta.lock() {
            if meta.remove(&request_id).is_some() {
                tracing::debug!(%request_id, "Removed metadata from pipeline_continue_meta");
            }
        }

        // Remove from pipeline_first_forward_done
        if let Ok(mut done) = self.pipeline_first_forward_done.lock() {
            // Remove all entries for this request_id (across all chunk_ids)
            done.retain(|(op_id, _)| *op_id != request_id);
        }
    }

}
