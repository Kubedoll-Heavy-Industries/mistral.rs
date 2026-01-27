//! Type-safe text generation pipeline.
//!
//! This module provides `TextPipeline<M>`, a typed pipeline where the model type
//! is known at compile time. This enables:
//! - **Monomorphization**: `model.transform()` calls are statically dispatched
//! - **Zero vtable overhead**: Hot paths have no dynamic dispatch
//! - **Type safety**: Invalid model types are caught at compile time
//!
//! # Architecture
//!
//! The typed pipeline differs from `GGUFPipeline` which uses `Box<dyn LanguageModel>`:
//!
//! | Aspect | `TextPipeline<M>` | `GGUFPipeline` |
//! |--------|-------------------|----------------|
//! | Model type | Known at compile time | Dynamic (`dyn`) |
//! | Forward dispatch | Static (monomorphized) | Virtual (vtable) |
//! | Engine usage | Via `Box<dyn Pipeline>` | Via `Box<dyn Pipeline>` |
//!
//! Both pipelines implement the `Pipeline` trait, so the engine can hold them
//! heterogeneously. The performance benefit comes from the forward pass being
//! statically dispatched within each pipeline.
//!
//! # Usage
//!
//! ```ignore
//! // Load a typed pipeline (architecture-specific)
//! let pipeline: TextPipeline<QLlama> = loader.load(device)?;
//!
//! // Forward pass is monomorphized - zero vtable overhead
//! let logits = pipeline.forward(&tokens, &ctx)?;
//! ```
//!
//! # Pipeline Parallelism
//!
//! The pipeline's role depends on hook configuration:
//! - **HEAD**: `embed()` → `transform()` → send activation
//! - **TAIL**: receive → `transform()` → `lm_head()`
//! - **FULL**: `embed()` → `transform()` → `lm_head()` (single node)

use std::any::Any;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::kv_cache::{CacheManager, FullCacheManager, NormalCache, NormalCacheManager};
use crate::models::{LanguageModel, PagedAttentionContext, TransformContext};
use crate::pipeline::hooks::{ActivationResult, HookContainer};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::text_models_inputs_processor::ModelInputs;
use crate::pipeline::{
    AnyMoePipelineMixin, CacheManagerMixin, ChatTemplate, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, MetadataMixin, ModelCategory, Pipeline, PreProcessingMixin,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;

/// Type-safe text generation pipeline.
///
/// Unlike `GGUFPipeline` which uses `Box<dyn LanguageModel>`, this pipeline
/// knows the model type `M` at compile time. The `forward()` method is
/// monomorphized, eliminating vtable overhead on the hot path.
///
/// # Type Parameter
///
/// `M: LanguageModel + Send + Sync` - The concrete model type (e.g., `QLlama`, `QQwen3`).
///
/// # Ownership
///
/// The pipeline owns:
/// - `model`: The language model (weights + architecture)
/// - `cache`: KV cache (working memory for generation)
/// - `tokenizer`, `chat_template`, `metadata`: Shared configuration
/// - `hook`: Optional transport for pipeline parallelism
pub struct TextPipeline<M: LanguageModel + Send + Sync> {
    /// The language model (weights + architecture).
    model: M,
    /// Tokenizer for encoding/decoding text.
    tokenizer: Arc<Tokenizer>,
    /// Chat template for message formatting.
    chat_template: Arc<ChatTemplate>,
    /// Model identifier (e.g., "meta-llama/Llama-3-8B").
    model_id: String,
    /// Device mapping for tensor placement.
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    /// Model metadata (max_seq_len, eos tokens, etc.).
    metadata: Arc<GeneralMetadata>,
    /// KV cache - working memory for generation.
    cache: EitherCache,
    /// Pipeline hook for distributed inference (PP stage communication).
    hook: Option<HookContainer>,
    /// Adapter registry for per-request LoRA adapter selection.
    adapter_registry: Option<Arc<crate::lora::AdapterRegistry>>,
}

impl<M: LanguageModel + Send + Sync> TextPipeline<M> {
    /// Create a new typed text pipeline.
    ///
    /// # Arguments
    ///
    /// * `model` - The language model
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `chat_template` - Chat template for message formatting
    /// * `model_id` - Model identifier
    /// * `mapper` - Device mapping
    /// * `metadata` - Model metadata
    pub fn new(
        model: M,
        tokenizer: Arc<Tokenizer>,
        chat_template: Arc<ChatTemplate>,
        model_id: String,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        metadata: Arc<GeneralMetadata>,
    ) -> Self {
        let num_layers = model.num_layers();
        let max_seq_len = model.max_seq_len();
        let cache = EitherCache::Normal(NormalCache::new(num_layers, max_seq_len));

        Self {
            model,
            tokenizer,
            chat_template,
            model_id,
            mapper,
            metadata,
            cache,
            hook: None,
            adapter_registry: None,
        }
    }

    /// Configure pipeline parallelism transport.
    pub fn with_hook(mut self, hook: HookContainer) -> Self {
        self.hook = Some(hook);
        self
    }

    /// Configure per-request LoRA adapter registry.
    pub fn with_adapter_registry(mut self, registry: Arc<crate::lora::AdapterRegistry>) -> Self {
        self.adapter_registry = Some(registry);
        self
    }

    /// Get the adapter registry if configured.
    pub fn adapter_registry(&self) -> Option<&Arc<crate::lora::AdapterRegistry>> {
        self.adapter_registry.as_ref()
    }

    /// Whether this stage has embedding weights (first stage).
    fn has_embedding(&self) -> bool {
        self.hook.as_ref().is_none_or(|h| h.is_first_stage())
    }

    /// Whether this stage has lm_head weights (last stage).
    fn has_lm_head(&self) -> bool {
        self.hook.as_ref().is_none_or(|h| h.is_last_stage())
    }

    /// Forward pass through the model.
    ///
    /// This is the **monomorphized hot path**. All model calls (`embed`, `transform`,
    /// `lm_head`) are statically dispatched since `M` is known at compile time.
    ///
    /// # Pipeline Parallelism
    ///
    /// The method handles all PP stages automatically:
    /// - **HEAD**: `embed()` → `transform()` → send activation, receive logits
    /// - **TAIL**: receive activation → `transform()` → `lm_head()`
    /// - **FULL**: `embed()` → `transform()` → `lm_head()`
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `seqlen_offsets` - Position offsets for RoPE
    /// * `context_lens` - (start, len) for extracting logits
    /// * `paged_attn_meta` - Optional paged attention metadata
    /// * `request_id` - Request UUID for PP coordination
    ///
    /// # Returns
    ///
    /// `ForwardInputsResult` - Either logits or PP coordination result
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        paged_attn_meta: Option<(
            Vec<(Tensor, Tensor)>,
            &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
        )>,
        request_id: uuid::Uuid,
    ) -> Result<ForwardInputsResult> {
        // Step 1: Get hidden states (embed OR receive from previous stage)
        let hidden = if self.has_embedding() {
            self.model.embed(input_ids)?
        } else {
            // TAIL: receive activation from previous stage
            match self.hook.as_ref().unwrap().receive_stage_input(request_id)? {
                Some(ActivationResult::Data { tensor }) => tensor,
                Some(ActivationResult::Completed { reason }) => {
                    return Ok(ForwardInputsResult::PipelineCompleted { reason });
                }
                None => {
                    candle_core::bail!("Non-first stage expected activation from previous stage");
                }
            }
        };

        // Step 2: Build context and transform
        let position_offset = if self.has_embedding() {
            seqlen_offsets.first().copied().unwrap_or(0)
        } else {
            // TAIL: position = cache length (tokens already processed)
            self.cache
                .normal()
                .0
                .first()
                .map_or(0, |c| c.current_seq_len())
        };

        let paged_attn_ctx = paged_attn_meta.as_ref().map(|(kv_cache, metadata)| {
            PagedAttentionContext {
                kv_cache: kv_cache.clone(),
                metadata,
            }
        });
        let ctx = TransformContext {
            seq_len: hidden.dims()[1],
            position_offset,
            paged_attn: paged_attn_ctx.as_ref(),
            flash_params: None,
            position_ids: None,
        };

        // Transform through layers - MONOMORPHIZED, zero vtable overhead
        let hidden = self
            .model
            .transform(hidden, &ctx, &mut self.cache.normal().0)?;

        // Step 3: Output (send activation OR compute logits)
        if !self.has_lm_head() {
            // HEAD: send activation to next stage
            self.hook
                .as_ref()
                .unwrap()
                .send_stage_output(&hidden, &[], request_id, position_offset)?;

            if self.hook.as_ref().unwrap().needs_external_logits() {
                // HEAD: wait for logits from TAIL
                let logits = self.hook.as_ref().unwrap().receive_response_logits()?;
                return Ok(ForwardInputsResult::CausalGeneration {
                    logits: logits.unsqueeze(1)?,
                });
            }

            candle_core::bail!("Middle stage completed send but has nowhere to return");
        }

        // TAIL or single-node: compute logits - MONOMORPHIZED
        let logits = self.model.lm_head(hidden)?;

        // Extract last position's logits for sampling
        let actual_context_lens = if !self.has_embedding() {
            // TAIL: extract only last position from actual logits shape
            let batch = logits.dims()[0];
            let seq_len = logits.dims()[1];
            vec![(seq_len - 1, 1); batch]
        } else {
            context_lens
        };

        let logits = crate::pipeline::extract_logits(&logits, actual_context_lens)?;
        Ok(ForwardInputsResult::RawLogits { logits })
    }

    /// Access the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Access the tokenizer.
    pub fn tokenizer_ref(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
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

impl<M: LanguageModel + Send + Sync + 'static> MetadataMixin for TextPipeline<M> {
    fn device(&self) -> Device {
        self.model.device().clone()
    }

    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }

    fn name(&self) -> String {
        self.model_id.clone()
    }

    fn reset_non_granular_state(&self) {
        // No non-granular state in typed pipeline
    }

    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }

    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

impl<M: LanguageModel + Send + Sync + 'static> CacheManagerMixin for TextPipeline<M> {
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

impl<M: LanguageModel + Send + Sync + 'static> PreProcessingMixin for TextPipeline<M> {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }

    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl<M: LanguageModel + Send + Sync + 'static> IsqPipelineMixin for TextPipeline<M> {
    fn re_isq_model(&mut self, _dtype: mistralrs_quant::IsqType) -> anyhow::Result<()> {
        anyhow::bail!("ISQ re-quantization not supported on typed pipeline")
    }
}

impl<M: LanguageModel + Send + Sync + 'static> AnyMoePipelineMixin for TextPipeline<M> {}

// ============================================================================
// Pipeline trait implementation
// ============================================================================

#[async_trait::async_trait]
impl<M: LanguageModel + Send + Sync + 'static> Pipeline for TextPipeline<M> {
    fn get_hook(&self) -> Option<&HookContainer> {
        self.hook.as_ref()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> std::result::Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full: _,
            seqlen_offsets,
            seqlen_offsets_full: _,
            context_lens,
            position_ids: _,
            paged_attn_meta,
            flash_meta: _,
            flash_meta_full: _,
            request_id,
            inference_step,
        } = *inputs.downcast().expect("Downcast failed.");

        // Set request context on hook BEFORE forward
        if let Some(hook) = self.hook.as_ref() {
            hook.set_request_context(request_id);

            use crate::pipeline::text_models_inputs_processor::InferenceStep;
            if let InferenceStep::Prefill {
                total_prompt_tokens,
                chunk_start_position,
                ..
            } = inference_step
            {
                hook.call_init_pipeline_request(
                    request_id,
                    total_prompt_tokens,
                    chunk_start_position,
                );
            }
        }

        // Get paged attention metadata if using paged attention
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            (Some(_), None) => {
                candle_core::bail!("Forward step expected PagedAttention metadata")
            }
            (None, Some(_)) => {
                candle_core::bail!("Got PagedAttention metadata but no cache engine")
            }
            (None, None) => None,
        };

        // Call the monomorphized forward method
        self.forward(
            &input_ids,
            &seqlen_offsets,
            context_lens,
            paged_attn_meta,
            request_id,
        )
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> std::result::Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }

    fn set_hook(&mut self, hook: HookContainer) {
        self.hook = Some(hook);
    }

    fn supports_hooks(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_pipeline_type_safety() {
        // This test verifies the type system - it should compile.
        // At runtime, we just check that the type exists.

        // The TextPipeline<M> type should be sized when M is known
        fn assert_sized<T: Sized>() {}
        // Can't call this without a concrete M, but the function itself compiles
        fn _verify_type_param<M: LanguageModel + Send + Sync + 'static>() {
            assert_sized::<TextPipeline<M>>();
        }
    }
}
