//! Monomorphized causal language model pipeline.
//!
//! `CausalLMPipeline` is an enum with a variant for each supported architecture.
//! All forward passes are monomorphized—no vtable dispatch in the hot path.
//!
//! # Design
//!
//! ```text
//! load_causal_lm_pipeline()
//!   → runtime architecture detection (cold path)
//!   → CausalLMPipeline::Llama(TextPipeline<QLlama>)    // monomorphized
//!   → CausalLMPipeline::Qwen3(TextPipeline<QQwen3>)    // monomorphized
//!   → ...
//! ```
//!
//! The enum dispatch at the top level is a single branch prediction.
//! All inner computation is statically dispatched.

use std::any::Any;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::lora::AdapterRegistry;
use crate::models::llama::LlamaModel;
use crate::models::mixtral::Mixtral;
use crate::models::quantized_mistral3::ModelWeights as QMistral3;
use crate::models::quantized_phi2::ModelWeights as QPhi2;
use crate::models::quantized_phi3::ModelWeights as QPhi3;
use crate::models::quantized_qwen::ModelWeights as QQwen;
use crate::models::quantized_qwen3::ModelWeights as QQwen3;
use crate::models::quantized_qwen3_moe::ModelWeights as QQwen3MoE;
use crate::models::quantized_starcoder2::ModelWeights as QStarcoder2;
use crate::models::smollm3::ModelWeights as SmolLm3;
use crate::pipeline::chat_template::ChatTemplate;
use crate::pipeline::hooks::HookContainer;
use crate::pipeline::text::TextPipeline;
use crate::pipeline::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, MetadataMixin, ModelCategory, Pipeline, PreProcessingMixin,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;

/// Monomorphized causal language model pipeline.
///
/// Each variant holds a `TextPipeline<M>` with a concrete model type.
/// All forward passes are statically dispatched—no vtable overhead.
///
/// # Supported Architectures
///
/// - `Llama` - LLaMA, LLaMA 2, LLaMA 3, Code Llama, etc. (dense models)
/// - `Mixtral` - Mixtral, Llama-MoE (Mixture of Experts)
/// - `Mistral3` - Mistral, Mistral Nemo
/// - `Phi2` - Phi-2
/// - `Phi3` - Phi-3, Phi-3.5
/// - `Qwen2` - Qwen 2
/// - `Qwen3` - Qwen 3
/// - `Qwen3MoE` - Qwen 3 MoE
/// - `Starcoder2` - StarCoder 2
/// - `SmolLm3` - SmolLM 3
pub enum CausalLMPipeline {
    Llama(TextPipeline<LlamaModel>),
    Mixtral(TextPipeline<Mixtral>),
    Mistral3(TextPipeline<QMistral3>),
    Phi2(TextPipeline<QPhi2>),
    Phi3(TextPipeline<QPhi3>),
    Qwen2(TextPipeline<QQwen>),
    Qwen3(TextPipeline<QQwen3>),
    Qwen3MoE(TextPipeline<QQwen3MoE>),
    Starcoder2(TextPipeline<QStarcoder2>),
    SmolLm3(TextPipeline<SmolLm3>),
}

/// Macro to dispatch a method call to the inner pipeline.
///
/// Usage: `dispatch!(self, p => p.method(args))`
macro_rules! dispatch {
    ($self:expr, $p:ident => $body:expr) => {
        match $self {
            CausalLMPipeline::Llama($p) => $body,
            CausalLMPipeline::Mixtral($p) => $body,
            CausalLMPipeline::Mistral3($p) => $body,
            CausalLMPipeline::Phi2($p) => $body,
            CausalLMPipeline::Phi3($p) => $body,
            CausalLMPipeline::Qwen2($p) => $body,
            CausalLMPipeline::Qwen3($p) => $body,
            CausalLMPipeline::Qwen3MoE($p) => $body,
            CausalLMPipeline::Starcoder2($p) => $body,
            CausalLMPipeline::SmolLm3($p) => $body,
        }
    };
}

// ============================================================================
// Trait implementations via dispatch macro
// ============================================================================

impl PreProcessingMixin for CausalLMPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        dispatch!(self, p => p.get_chat_template())
    }

    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        dispatch!(self, p => p.get_input_processor_config())
    }
}

impl IsqPipelineMixin for CausalLMPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> anyhow::Result<()> {
        dispatch!(self, p => p.re_isq_model(dtype))
    }
}

impl CacheManagerMixin for CausalLMPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        dispatch!(self, p => p.clone_in_cache(seqs))
    }

    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        dispatch!(self, p => p.clone_out_cache(seqs))
    }

    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        dispatch!(self, p => p.set_none_cache(seqs, reset_non_granular, modify_draft_cache, load_preallocated_cache))
    }

    fn cache(&self) -> &EitherCache {
        dispatch!(self, p => p.cache())
    }
}

impl MetadataMixin for CausalLMPipeline {
    fn device(&self) -> Device {
        dispatch!(self, p => p.device())
    }

    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        dispatch!(self, p => p.tokenizer())
    }

    fn name(&self) -> String {
        dispatch!(self, p => p.name())
    }

    fn reset_non_granular_state(&self) {
        dispatch!(self, p => p.reset_non_granular_state())
    }

    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        dispatch!(self, p => p.get_metadata())
    }

    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        dispatch!(self, p => p.device_mapper())
    }
}

// AnyMoePipelineMixin - use default implementations (not supported for CausalLMPipeline)
impl AnyMoePipelineMixin for CausalLMPipeline {}

#[async_trait::async_trait]
impl Pipeline for CausalLMPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        dispatch!(self, p => p.forward_inputs(inputs, return_raw_logits))
    }

    fn get_hook(&self) -> Option<&HookContainer> {
        dispatch!(self, p => p.get_hook())
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        dispatch!(self, p => p.sample_causal_gen(seqs, logits, prefix_cacher, disable_eos_stop, rng).await)
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }

    fn set_hook(&mut self, hook: HookContainer) {
        dispatch!(self, p => p.set_hook(hook))
    }

    fn supports_hooks(&self) -> bool {
        true
    }
}

impl CausalLMPipeline {
    /// Get the architecture name for this pipeline.
    pub fn architecture(&self) -> &'static str {
        match self {
            Self::Llama(_) => "Llama",
            Self::Mixtral(_) => "Mixtral",
            Self::Mistral3(_) => "Mistral3",
            Self::Phi2(_) => "Phi2",
            Self::Phi3(_) => "Phi3",
            Self::Qwen2(_) => "Qwen2",
            Self::Qwen3(_) => "Qwen3",
            Self::Qwen3MoE(_) => "Qwen3MoE",
            Self::Starcoder2(_) => "Starcoder2",
            Self::SmolLm3(_) => "SmolLm3",
        }
    }

    /// Set the adapter registry for per-request LoRA adapter selection.
    ///
    /// The registry allows runtime adapter switching without rebuilding the model.
    /// When requests include adapter names, they will be activated via this registry
    /// before each forward pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[path])
    ///     .build()?;
    ///
    /// // Create and populate registry
    /// let registry = AdapterRegistry::new(pipeline.device());
    /// registry.register("style", config, weights)?;
    ///
    /// // Attach registry to pipeline
    /// pipeline.set_adapter_registry(Arc::new(registry));
    /// ```
    pub fn set_adapter_registry(&mut self, registry: Arc<AdapterRegistry>) {
        dispatch!(self, p => p.set_adapter_registry(registry))
    }

    /// Get the adapter registry if configured.
    pub fn adapter_registry(&self) -> Option<&Arc<AdapterRegistry>> {
        dispatch!(self, p => p.adapter_registry())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_lm_pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CausalLMPipeline>();
    }
}
