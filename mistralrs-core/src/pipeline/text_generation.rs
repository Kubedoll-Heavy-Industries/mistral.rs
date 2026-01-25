//! Unified text generation pipeline.
//!
//! This pipeline handles autoregressive text generation for any model that implements
//! `TransformerModel`, regardless of serialization format (GGUF, safetensors, ONNX).
//!
//! # Design
//!
//! The pipeline is mostly **immutable configuration**:
//! - Model (weights + tokenizer + config)
//! - Hooks (transport wiring for pipeline parallelism)
//! - Device mapping
//!
//! The only mutable state is the **KV cache**, which is working memory for generation.
//!
//! # Pipeline Parallelism
//!
//! The pipeline's role depends on what weights were loaded:
//! - **HEAD**: has embedding → embed, transform, send
//! - **MIDDLE**: no embedding/lm_head → receive, transform, send
//! - **TAIL**: has lm_head → receive, transform, lm_head
//! - **FULL**: all weights → embed, transform, lm_head (single node)

use std::any::Any;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::kv_cache::{FullCacheManager, NormalCacheManager};
use crate::models::{PagedAttentionContext, TransformContext, TransformerModel};
use crate::paged_attention::ModelConfigMetadata;
use crate::pipeline::hooks::{ActivationResult, HookContainer};
use crate::pipeline::{
    AnyMoePipelineMixin, CacheManager, CacheManagerMixin, EitherCache,
    GeneralMetadata, IsqPipelineMixin, MetadataMixin, PreProcessingMixin,
};
use crate::sequence::Sequence;

/// Input to a pipeline stage.
#[derive(Debug)]
pub enum StageInput {
    /// First stage: raw token IDs to embed.
    Tokens(Tensor),
    /// Middle/tail stage: activation received from previous stage.
    Activation(Tensor),
}

/// Output from a pipeline stage.
#[derive(Debug)]
pub enum StageOutput {
    /// Last stage: logits for sampling.
    Logits(Tensor),
    /// Non-last stage: activation was sent to next stage.
    Sent,
    /// Pipeline completed (received completion signal).
    Completed(crate::StopReason),
}

/// Configuration that's immutable after loading.
pub struct PipelineConfig {
    /// The transformer model (weights + architecture).
    pub model: Arc<dyn TransformerModel + Send + Sync>,
    /// Tokenizer for encoding/decoding text.
    pub tokenizer: Arc<Tokenizer>,
    /// Model identifier (e.g., "meta-llama/Llama-3-8B").
    pub model_id: String,
    /// Device mapping for tensor placement.
    pub mapper: Arc<dyn DeviceMapper + Send + Sync>,
    /// Model metadata (max_seq_len, eos tokens, etc.).
    pub metadata: Arc<GeneralMetadata>,
    /// Primary device for this pipeline.
    pub device: Device,
    /// Paged attention config, if enabled.
    pub paged_attn_config: Option<ModelConfigMetadata>,
}

/// Transport configuration for pipeline parallelism.
pub struct TransportConfig {
    /// Hook container for PP communication.
    pub hooks: HookContainer,
}

/// Unified text generation pipeline.
///
/// Handles autoregressive generation for any `TransformerModel`, with optional
/// pipeline parallelism via hooks.
pub struct TextGenerationPipeline {
    // === Immutable configuration ===
    config: PipelineConfig,
    /// Transport hooks for pipeline parallelism (None = single node).
    transport: Option<TransportConfig>,

    // === Mutable runtime state ===
    /// KV cache - the only mutable state.
    cache: EitherCache,
}

impl TextGenerationPipeline {
    /// Create a new pipeline from loaded components.
    pub fn new(config: PipelineConfig, cache: EitherCache) -> Self {
        Self {
            config,
            transport: None,
            cache,
        }
    }

    /// Configure pipeline parallelism transport.
    pub fn with_transport(mut self, transport: TransportConfig) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Whether this stage should embed tokens (has embedding weights).
    fn has_embedding(&self) -> bool {
        // For now, assume all loaded models can embed.
        // Future: check if embedding weights are present.
        true
    }

    /// Whether this stage should compute logits (has lm_head weights).
    fn has_lm_head(&self) -> bool {
        // Determined by hooks: if we have hooks and aren't last stage, we don't compute logits.
        match &self.transport {
            Some(t) => t.hooks.is_last_stage(),
            None => true, // Single node always computes logits
        }
    }

    /// Whether this stage receives activations from a previous stage.
    fn receives_activation(&self) -> bool {
        match &self.transport {
            Some(t) => !t.hooks.is_first_stage(),
            None => false,
        }
    }

    /// Forward pass through the pipeline.
    ///
    /// Handles the full PP flow:
    /// - HEAD: embed → transform → send
    /// - MIDDLE: receive → transform → send
    /// - TAIL: receive → transform → lm_head
    /// - FULL: embed → transform → lm_head
    pub fn forward(
        &self,
        tokens: &Tensor,
        position_offset: usize,
        request_id: uuid::Uuid,
        paged_attn_meta: Option<(&[(Tensor, Tensor)], &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata)>,
    ) -> Result<StageOutput> {
        // Step 1: Get input (embed tokens or receive activation)
        let hidden = if self.receives_activation() {
            let hooks = &self.transport.as_ref().unwrap().hooks;
            match hooks.receive_stage_input(request_id)? {
                Some(ActivationResult::Data { tensor }) => tensor,
                Some(ActivationResult::Completed { reason }) => {
                    return Ok(StageOutput::Completed(reason));
                }
                None => {
                    // First stage but configured to receive? Fall back to embedding.
                    self.config.model.embed(tokens)?
                }
            }
        } else {
            self.config.model.embed(tokens)?
        };

        // Step 2: Build transform context
        let paged_attn_ctx = paged_attn_meta.map(|(kv_cache, metadata)| {
            PagedAttentionContext {
                kv_cache: kv_cache.to_vec(),
                metadata
            }
        });
        let ctx = TransformContext {
            seq_len: hidden.dims()[1],
            position_offset,
            paged_attn: paged_attn_ctx.as_ref(),
        };

        // Step 3: Transform through our layers
        let hidden = {
            let mut cache_guard = self.cache.normal();
            self.config.model.transform(hidden, &ctx, &mut cache_guard.0)?
        };

        // Step 4: Output (send activation or compute logits)
        if !self.has_lm_head() {
            // HEAD/MIDDLE: send to next stage
            let hooks = &self.transport.as_ref().unwrap().hooks;
            let toks: Vec<u32> = tokens.flatten_all()?.to_vec1()?;
            hooks.send_stage_output(&hidden, &toks, request_id, position_offset)?;

            if hooks.needs_external_logits() {
                // HEAD stage: wait for logits from TAIL
                let logits = hooks.receive_response_logits()?;
                Ok(StageOutput::Logits(logits))
            } else {
                Ok(StageOutput::Sent)
            }
        } else {
            // TAIL/FULL: compute logits
            let logits = self.config.model.lm_head(hidden)?;
            Ok(StageOutput::Logits(logits))
        }
    }

    /// Access the model.
    pub fn model(&self) -> &Arc<dyn TransformerModel + Send + Sync> {
        &self.config.model
    }

    /// Access the tokenizer.
    pub fn tokenizer_ref(&self) -> &Arc<Tokenizer> {
        &self.config.tokenizer
    }

    /// Access the cache.
    pub fn cache(&self) -> &EitherCache {
        &self.cache
    }

    /// Access the cache mutably.
    pub fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
}

// ============================================================================
// Pipeline trait implementations
// ============================================================================

impl MetadataMixin for TextGenerationPipeline {
    fn device(&self) -> Device {
        self.config.device.clone()
    }

    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.config.tokenizer.clone())
    }

    fn name(&self) -> String {
        self.config.model_id.clone()
    }

    fn reset_non_granular_state(&self) {
        // No non-granular state in unified pipeline
    }

    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.config.metadata.clone()
    }

    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(self.config.mapper.as_ref())
    }
}

impl CacheManagerMixin for TextGenerationPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache, EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, false)
        }
    }

    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache, EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, false)
        }
    }

    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if matches!(self.cache, EitherCache::Full(_)) {
            FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        } else {
            NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            );
        }
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

impl PreProcessingMixin for TextGenerationPipeline {
    fn get_chat_template(&self) -> Option<Arc<crate::pipeline::ChatTemplate>> {
        // Chat template is optional - may be None for raw completion
        // TODO: Add chat_template to PipelineConfig if needed for chat-style pipelines
        None
    }

    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for TextGenerationPipeline {
    fn re_isq_model(&mut self, _dtype: mistralrs_quant::IsqType) -> anyhow::Result<()> {
        // ISQ is a loading concern, not runtime
        anyhow::bail!("ISQ re-quantization not supported on unified pipeline")
    }
}

impl AnyMoePipelineMixin for TextGenerationPipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_input_variants() {
        // Just verify the types compile correctly
        let _tokens = StageInput::Tokens(
            Tensor::zeros(&[1, 10], candle_core::DType::U32, &Device::Cpu).unwrap()
        );
        let _activation = StageInput::Activation(
            Tensor::zeros(&[1, 10, 512], candle_core::DType::F32, &Device::Cpu).unwrap()
        );
    }
}
