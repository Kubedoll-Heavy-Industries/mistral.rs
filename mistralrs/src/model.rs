use anyhow::Context;
use candle_core::{Device, Result, Tensor};
use either::Either;
use futures::future::join_all;
use mistralrs_core::*;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver};

use crate::{Chat, EmbeddingRequest, EmbeddingRequestBuilder, TextMessages};

/// Gets the best device, cpu, cuda if compiled with CUDA, or Metal
pub fn best_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

/// The object used to interact with the model. This can be used with many varietes of models, \
/// and as such may be created with one of:
/// - [`TextModelBuilder`]
/// - [`LoraModelBuilder`]
/// - [`XLoraModelBuilder`]
/// - [`GgufModelBuilder`]
/// - [`GgufLoraModelBuilder`]
/// - [`GgufXLoraModelBuilder`]
/// - [`VisionModelBuilder`]
/// - [`AnyMoeModelBuilder`]
///
/// [`TextModelBuilder`]: crate::TextModelBuilder
/// [`LoraModelBuilder`]: crate::LoraModelBuilder
/// [`XLoraModelBuilder`]: crate::XLoraModelBuilder
/// [`GgufModelBuilder`]: crate::GgufModelBuilder
/// [`GgufModelBuilder`]: crate::GgufModelBuilder
/// [`GgufLoraModelBuilder`]: crate::GgufLoraModelBuilder
/// [`GgufXLoraModelBuilder`]: crate::GgufXLoraModelBuilder
/// [`VisionModelBuilder`]: crate::VisionModelBuilder
/// [`AnyMoeModelBuilder`]: crate::AnyMoeModelBuilder
///
pub struct Model {
    pub(crate) runner: Arc<MistralRs>,
}

pub struct Stream<'a> {
    _server: &'a Model,
    rx: Receiver<Response>,
}

impl Stream<'_> {
    pub async fn next(&mut self) -> Option<Response> {
        self.rx.recv().await
    }
}

impl Model {
    pub fn new(runner: Arc<MistralRs>) -> Self {
        Self { runner }
    }

    /// Stream a chat request.
    pub async fn stream_chat_request(&self, request: Chat) -> anyhow::Result<Stream<'_>> {
        let (tx, rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: uuid::Uuid::nil(),
            input: request.into_inference_input(true),
            response: tx,
            model_id: None,
        }));

        self.runner.get_sender(None)?.send(request).await?;

        Ok(Stream { _server: self, rx })
    }

    /// Send a chat request (non-streaming).
    pub async fn send_chat_request(&self, request: Chat) -> anyhow::Result<ChatCompletionResponse> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: uuid::Uuid::nil(),
            input: request.into_inference_input(false),
            response: tx,
            model_id: None,
        }));

        self.runner.get_sender(None)?.send(request).await?;

        let ResponseOk::Done(response) = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok(response)
    }

    /// Generate with the model, returning raw logits of the first token generated.
    ///
    /// Returns the chunks of the logits (1 or more, determined by prompt batchsize) and the tokens.
    pub async fn send_raw_chat_request(
        &self,
        mut request: Chat,
    ) -> anyhow::Result<(Vec<Tensor>, Vec<u32>)> {
        let (tx, mut rx) = channel(1);

        request.return_raw_logits = true;

        let request = Request::Normal(Box::new(NormalRequest {
            id: uuid::Uuid::nil(),
            input: request.into_inference_input(false),
            response: tx,
            model_id: None,
        }));

        self.runner.get_sender(None)?.send(request).await?;

        let ResponseOk::Raw {
            logits_chunks,
            tokens,
        } = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok((logits_chunks, tokens))
    }

    pub async fn generate_image(
        &self,
        prompt: impl ToString,
        response_format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: uuid::Uuid::nil(),
            input: mistralrs_core::InferenceInput {
                op: mistralrs_core::InferenceOperation::ImageGeneration {
                    prompt: prompt.to_string(),
                    format: response_format,
                    generation_params,
                },
                exec: mistralrs_core::InferenceExec {
                    is_streaming: false,
                    truncate_sequence: false,
                },
                adapters: None,
            },
            response: tx,
            model_id: None,
        }));

        self.runner.get_sender(None)?.send(request).await?;

        let ResponseOk::ImageGeneration(response) = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok(response)
    }

    /// Generate audio given a (model specific) prompt.
    ///
    /// This returns: (pcm, sampling rate, channels)
    pub async fn generate_speech(
        &self,
        prompt: impl ToString,
    ) -> anyhow::Result<(Arc<Vec<f32>>, usize, usize)> {
        let (tx, mut rx) = channel(1);

        let request = Request::Normal(Box::new(NormalRequest {
            id: uuid::Uuid::nil(),
            input: mistralrs_core::InferenceInput {
                op: mistralrs_core::InferenceOperation::SpeechGeneration {
                    prompt: prompt.to_string(),
                },
                exec: mistralrs_core::InferenceExec {
                    is_streaming: false,
                    truncate_sequence: false,
                },
                adapters: None,
            },
            response: tx,
            model_id: None,
        }));

        self.runner.get_sender(None)?.send(request).await?;

        let ResponseOk::Speech {
            pcm,
            rate,
            channels,
        } = rx
            .recv()
            .await
            .context("Channel was erroneously closed!")?
            .as_result()?
        else {
            anyhow::bail!("Got unexpected response type.")
        };

        Ok((pcm, rate, channels))
    }

    /// Generate embeddings for one or more inputs configured via an [`EmbeddingRequestBuilder`].
    ///
    /// Returns one embedding vector per input in the same order they were added.
    pub async fn generate_embeddings(
        &self,
        request: EmbeddingRequestBuilder,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let request = request.build()?;
        let EmbeddingRequest {
            inputs,
            truncate_sequence,
        } = request;

        let runner = self.runner.clone();
        let futures = inputs.into_iter().map(|input| {
            let runner = runner.clone();
            async move {
                let op = input.into_operation();
                let (tx, mut rx) = channel(1);

                let request = Request::Normal(Box::new(NormalRequest {
                    id: uuid::Uuid::nil(),
                    input: mistralrs_core::InferenceInput {
                        op,
                        exec: mistralrs_core::InferenceExec {
                            is_streaming: false,
                            truncate_sequence,
                        },
                        adapters: None,
                    },
                    response: tx,
                    model_id: None,
                }));

                runner
                    .get_sender(None)?
                    .send(request)
                    .await
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

                let ResponseOk::Embeddings { embeddings, .. } = rx
                    .recv()
                    .await
                    .context("Channel was erroneously closed!")?
                    .as_result()?
                else {
                    anyhow::bail!("Got unexpected response type.")
                };

                Ok::<Vec<f32>, anyhow::Error>(embeddings)
            }
        });

        let results = join_all(futures).await;
        let mut embeddings = Vec::with_capacity(results.len());
        for result in results {
            embeddings.push(result?);
        }
        Ok(embeddings)
    }

    /// Convenience wrapper for generating a single embedding.
    pub async fn generate_embedding(&self, prompt: impl ToString) -> anyhow::Result<Vec<f32>> {
        let mut embeddings = self
            .generate_embeddings(EmbeddingRequest::builder().add_prompt(prompt.to_string()))
            .await?;

        Ok(embeddings
            .pop()
            .expect("EmbeddingRequestBuilder should guarantee at least one input"))
    }

    /// Reapply ISQ to the model. This will be done on whatever device the model is already on.
    pub async fn re_isq_model(&self, isq_type: IsqType) -> anyhow::Result<()> {
        let request = Request::ReIsq(isq_type);

        Ok(self.runner.get_sender(None)?.send(request).await?)
    }

    /// Tokenize some text or messages.
    /// - `tools` is only used if messages are provided.
    pub async fn tokenize(
        &self,
        text: Either<TextMessages, String>,
        tools: Option<Vec<Tool>>,
        add_special_tokens: bool,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<Vec<u32>> {
        let (tx, mut rx) = channel(1);
        let request = Request::Tokenize(TokenizationRequest {
            id: uuid::Uuid::nil(),
            input: TokenizeInput {
                text: text.map_left(Into::into),
                tools,
                add_generation_prompt,
                add_special_tokens,
                enable_thinking,
                reasoning_effort: None,
            },
            response: tx,
            model_id: None,
        });
        self.runner.get_sender(None)?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    /// Detokenize some tokens.
    pub async fn detokenize(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
    ) -> anyhow::Result<String> {
        let (tx, mut rx) = channel(1);
        let request = Request::Detokenize(DetokenizationRequest {
            id: uuid::Uuid::nil(),
            input: DetokenizeInput {
                tokens,
                skip_special_tokens,
            },
            response: tx,
            model_id: None,
        });
        self.runner.get_sender(None)?.send(request).await?;

        rx.recv().await.context("Channel was erroneously closed!")?
    }

    /// Retrieve some information about this model.
    pub fn config(&self) -> std::result::Result<MistralRsConfig, String> {
        self.runner.config(None)
    }

    /// Returns the maximum supported sequence length for this model, if applicable.
    pub fn max_sequence_length(&self) -> std::result::Result<Option<usize>, MistralRsError> {
        self.runner.max_sequence_length(None)
    }

    pub fn inner(&self) -> &MistralRs {
        &self.runner
    }
}
