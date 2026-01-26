use super::llg::build_llg_factory;
use super::{
    get_model_paths, get_xlora_paths,
    text_models_inputs_processor::ModelInputs,
    AdapterKind, CacheManager, GeneralMetadata, HookContainer, Loader, ModelKind, ModelPaths,
    PrettyName, QuantizationKind, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::models::{LanguageModel, PagedAttentionContext, TransformContext};
use crate::pipeline::hooks::ActivationResult;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::gguf::{
    get_gguf_chat_template, {convert_gguf_to_hf_tokenizer, GgufTokenizerConversion},
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::kv_cache::{FullCacheManager, NormalCache, NormalCacheManager};
use crate::lora::Ordering;
use crate::paged_attention::{
    calculate_cache_config, AttentionImplementation, CacheEngine, ModelConfigLike,
    ModelConfigMetadata,
};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::loaders::DeviceMappedModelLoader;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths_gguf, DeviceMapSetting, LocalModelPaths, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_mistral3::ModelWeights as QMistral3,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_qwen::ModelWeights as QQwen,
    models::quantized_qwen3::ModelWeights as QQwen3,
    models::quantized_qwen3_moe::ModelWeights as QQwen3MoE,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use either::Either;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Model variant: either a TransformerModel (unified path) or XLora (legacy two-pass).
///
/// This replaces the old 10-variant enum with a simpler structure:
/// - LanguageModel implementations use dynamic dispatch (Box<dyn>)
/// - XLora models stay concrete until we create an XLoraModel trait
enum ModelVariant {
    /// Models implementing LanguageModel (stateless, unified forward path)
    Transformer(Box<dyn LanguageModel + Send + Sync>),
    /// XLora Llama (two-pass inference, owns cache)
    XLoraLlama(XLoraQLlama),
    /// XLora Phi3 (two-pass inference, owns cache)
    XLoraPhi3(XLoraQPhi3),
}

impl ModelVariant {
    /// Get this model as a LanguageModel trait object, if it implements the trait.
    ///
    /// Returns Some for LanguageModel, None for XLora models (which need two-pass inference).
    fn as_language_model(&self) -> Option<&dyn LanguageModel> {
        match self {
            ModelVariant::Transformer(model) => Some(&**model),
            ModelVariant::XLoraLlama(_) | ModelVariant::XLoraPhi3(_) => None,
        }
    }

    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize {
        match self {
            ModelVariant::Transformer(model) => model.num_layers(),
            ModelVariant::XLoraLlama(model) => model.cache.normal().0.len(),
            ModelVariant::XLoraPhi3(model) => model.cache.normal().0.len(),
        }
    }

    /// Maximum sequence length supported by this model.
    fn max_seq_len(&self) -> usize {
        match self {
            ModelVariant::Transformer(model) => model.max_seq_len(),
            ModelVariant::XLoraLlama(model) => model.max_seq_len,
            ModelVariant::XLoraPhi3(model) => model.max_seq_len,
        }
    }

    /// The device where model weights reside.
    fn device(&self) -> &Device {
        match self {
            ModelVariant::Transformer(model) => model.device(),
            ModelVariant::XLoraLlama(model) => &model.device,
            ModelVariant::XLoraPhi3(model) => &model.device,
        }
    }
}


/// Pipeline for GGUF models using runtime polymorphism.
///
/// For better type safety and monomorphized hot paths, use `TextPipeline<M>`
/// with typed loaders instead.
#[deprecated(
    since = "0.8.0",
    note = "Use TextPipeline<M> with typed loaders (e.g., CausalLMLoader) instead. \
            GGUFPipeline uses runtime polymorphism; TextPipeline provides \
            compile-time type safety and monomorphized inference paths."
)]
pub struct GGUFPipeline {
    model: ModelVariant,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    /// KV cache owned by the pipeline (models are stateless)
    cache: EitherCache,
    /// Pipeline hook for distributed inference (PP stage communication)
    hook: Option<HookContainer>,
}

/// Unified forward pass for models implementing TransformerModel.
///
/// This is the ONE code path for inference. Pipeline parallelism is just
/// configuration - hooks fire at boundaries if layers are distributed,
/// otherwise everything runs locally.
///
/// Architecture:
/// - HEAD (first stage): embed() → transform() → send activation
/// - TAIL (last stage):  receive → transform() → lm_head()
/// - Single node:        embed() → transform() → lm_head()
///
/// Key insight: After embed(), tokens are gone. Only hidden states flow.
///
/// This is a free function to allow borrowing model, cache, and hook separately
/// (avoids borrow checker conflicts with &mut self).
#[allow(clippy::too_many_arguments)]
fn forward_transformer(
    model: &dyn LanguageModel,
    cache: &mut [crate::kv_cache::KvCache],
    hook: Option<&HookContainer>,
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
    context_lens: Vec<(usize, usize)>,
    paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata)>,
    request_id: uuid::Uuid,
) -> candle_core::Result<ForwardInputsResult> {
    let has_embedding = hook.as_ref().is_none_or(|h| h.is_first_stage());
    let has_lm_head = hook.as_ref().is_none_or(|h| h.is_last_stage());

    // Step 1: Get hidden states (embed OR receive from previous stage)
    let hidden = if has_embedding {
        model.embed(input_ids)?
    } else {
        // TAIL/MIDDLE: receive activation from previous stage
        match hook.as_ref().unwrap().receive_stage_input(request_id)? {
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
    // For TAIL, derive position from cache (stream cursor model)
    let position_offset = if has_embedding {
        seqlen_offsets.first().copied().unwrap_or(0)
    } else {
        // TAIL: position = cache length (tokens already processed)
        cache.first().map_or(0, |c| c.current_seq_len())
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
    };

    let hidden = model.transform(hidden, &ctx, cache)?;

    // Step 3: Output (send activation OR compute logits)
    if !has_lm_head {
        // HEAD/MIDDLE: send activation to next stage
        // Note: tokens are not used by streaming_hook - only hidden states matter
        hook.as_ref().unwrap().send_stage_output(&hidden, &[], request_id, position_offset)?;

        if hook.as_ref().unwrap().needs_external_logits() {
            // HEAD: wait for logits from TAIL
            let logits = hook.as_ref().unwrap().receive_response_logits()?;
            return Ok(ForwardInputsResult::CausalGeneration {
                logits: logits.unsqueeze(1)?,
            });
        }

        // MIDDLE: shouldn't reach here (has no lm_head but doesn't need external logits)
        candle_core::bail!("Middle stage completed send but has nowhere to return");
    }

    // TAIL or single-node: compute logits
    let logits = model.lm_head(hidden)?;

    // Extract last position's logits for sampling.
    // For TAIL, context_lens was computed for HEAD's tokens - use logits shape instead.
    let actual_context_lens = if !has_embedding {
        // TAIL: extract only last position from actual logits shape
        let batch = logits.dims()[0];
        let seq_len = logits.dims()[1];
        vec![(seq_len - 1, 1); batch]
    } else {
        // Single-node or HEAD with lm_head: use passed-in context_lens
        context_lens
    };

    let logits = crate::pipeline::extract_logits(&logits, actual_context_lens)?;
    Ok(ForwardInputsResult::RawLogits { logits })
}

/// Loader for a GGUF model.
///
/// This loader uses runtime polymorphism for model dispatch. For better type safety
/// and monomorphized hot paths, use typed loaders instead:
/// - `CausalLMLoader` for text generation models
/// - (Future: `EmbeddingLoader`, `VisionLoader`, etc.)
#[deprecated(
    since = "0.8.0",
    note = "Use typed loaders instead: CausalLMLoader for text models. \
            GGUFLoader uses runtime polymorphism; typed loaders provide \
            compile-time type safety and monomorphized inference paths."
)]
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
#[deprecated(
    since = "0.8.0",
    note = "Use typed loader builders instead (e.g., CausalLMLoaderBuilder). \
            Configuration is passed directly to typed loaders."
)]
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
    /// Layer range for pipeline parallelism.
    /// Only loads layers in this range, enabling distributed inference.
    pub layer_range: Option<std::ops::Range<usize>>,
}

#[derive(Default)]
/// A builder for a GGUF loader.
#[deprecated(
    since = "0.8.0",
    note = "Use typed loader builders instead: CausalLMLoaderBuilder for text models. \
            Typed builders provide better ergonomics and type safety."
)]
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
}

#[allow(deprecated)]
impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` is the model ID where you can find a
    /// `tokenizer_config.json` file. If the `chat_template` is specified, then it will be treated as a
    /// path and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
            lora_adapter_ids: None,
        })
    }
}

#[allow(deprecated)]
impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            kind,
            tgt_non_granular_index,
            config,
            jinja_explicit,
            lora_adapter_ids: None,
        }
    }
}

#[allow(deprecated)]
impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent
        );
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }

        info!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers = readers.iter_mut().collect::<Vec<_>>();

        let model = Content::from_readers(&mut readers)?;
        if !silent {
            model.print_metadata()?;
        }
        let arch = model.arch();

        // If auto, convert to Map
        let total_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        // For pipeline parallelism, compute device map for only the loaded layers
        let (num_layers, layer_range_for_sizes) = match &self.config.layer_range {
            Some(range) => {
                let end = range.end.min(total_layers);
                let start = range.start.min(end);
                info!(
                    "Pipeline parallelism: computing device map for {} layers (range {:?} of {} total)",
                    end - start, range, total_layers
                );
                (end - start, Some(start..end))
            }
            None => (total_layers, None),
        };

        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let devices = device_map::get_all_similar_devices(device)?;
            // Initial dtype
            let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

            let model = GgufDeviceMapLoaderInner {
                model: &model,
                arch,
            };

            // Get layer sizes, slicing to loaded range for pipeline parallelism
            let all_layer_sizes =
                model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let layer_sizes_in_bytes = match &layer_range_for_sizes {
                Some(range) => all_layer_sizes[range.clone()].to_vec(),
                None => all_layer_sizes,
            };

            // For pipeline parallelism, only include non-mapped tensors for stages that have them:
            // - First stage (layer 0): token_embd
            // - Last stage (final layer): output_norm + output/lm_head
            // - Middle stages: neither
            let non_mapped_size_in_bytes = match &self.config.layer_range {
                Some(range) => {
                    let is_first_stage = range.start == 0;
                    let is_last_stage = range.end >= total_layers;

                    let mut size = 0usize;

                    // First stage needs token embeddings
                    if is_first_stage {
                        if let Ok(t) = model.model.tensor_info("token_embd.weight") {
                            // Embeddings are dequantized to F32 at runtime
                            size += t.shape.elem_count() * DType::F32.size_in_bytes();
                        }
                    }

                    // Last stage needs output_norm and lm_head
                    if is_last_stage {
                        if let Ok(t) = model.model.tensor_info("output_norm.weight") {
                            size += t.shape.elem_count() * DType::F32.size_in_bytes();
                        }
                        // output.weight (lm_head) - may be tied to token_embd
                        if model.model.has_tensor("output.weight") {
                            if let Ok(t) = model.model.tensor_info("output.weight") {
                                size += t.shape.elem_count() / t.ggml_dtype.block_size() * t.ggml_dtype.type_size();
                            }
                        } else if let Ok(t) = model.model.tensor_info("token_embd.weight") {
                            // Tied embeddings - reuse token_embd for output
                            size += t.shape.elem_count() / t.ggml_dtype.block_size() * t.ggml_dtype.type_size();
                        }
                    }

                    tracing::debug!(
                        is_first_stage,
                        is_last_stage,
                        non_mapped_size_mib = size / (1024 * 1024),
                        "Pipeline stage non-mapped size (embed/lm_head)"
                    );
                    size
                }
                None => {
                    // No pipeline parallelism - include all non-mapped tensors
                    model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None)?
                }
            };
            let total_model_size_in_bytes =
                layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

            let new = model.get_device_layers(
                "this is a dummy config!",
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                paged_attn_config.as_ref(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }

        let pipeline_mapper =
            mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;
        let mapper = mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;
        let mut layer_devices = Vec::new();
        for layer in 0..num_layers {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu {
            // For pipeline parallelism, fail if layers don't fit on GPU.
            // CPU offloading doesn't work with distributed pipeline execution.
            if let Some(ref range) = self.config.layer_range {
                anyhow::bail!(
                    "Pipeline parallelism: layers {}..{} don't fit on available GPU(s). \
                     The coordinator assigned {} layers to this node, but they require CPU offloading. \
                     Either increase GPU memory, reduce model size, or adjust pipeline stage splits.",
                    range.start,
                    range.end.min(total_layers),
                    num_layers
                );
            }
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&model)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Only load gguf chat template if there is nothing else
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template(&model)?
            } else {
                None
            };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let paged_attn_config = if matches!(self.kind, ModelKind::GgufAdapter { .. }) {
            warn!("Adapter models do not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let model_config_metadata: ContentConfig = (&model).into();
        let internal_dtype = mapper.get_min_dtype(dtype)?;

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGUF(
                model,
                (device, mapper).into(),
                if paged_attn_config.is_some() {
                    AttentionImplementation::PagedAttention
                } else {
                    AttentionImplementation::Eager
                },
                internal_dtype,
                self.config.layer_range.clone(),  // Pipeline parallelism layer range
            );

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::new(quant, adapter)
        };

        // Config into model:
        // TransformerModel implementations are boxed for dynamic dispatch.
        // XLora models stay concrete (they have different forward signature).
        let model: ModelVariant = match self.kind {
            ModelKind::GgufQuantized { .. } => match arch {
                GGUFArchitecture::Llama => {
                    ModelVariant::Transformer(Box::new(QLlama::try_from(model_config)?))
                }
                GGUFArchitecture::Mistral3 => {
                    ModelVariant::Transformer(Box::new(QMistral3::try_from(model_config)?))
                }
                GGUFArchitecture::Phi2 => {
                    ModelVariant::Transformer(Box::new(QPhi::try_from(model_config)?))
                }
                GGUFArchitecture::Phi3 => {
                    ModelVariant::Transformer(Box::new(QPhi3::try_from(model_config)?))
                }
                GGUFArchitecture::Starcoder2 => {
                    ModelVariant::Transformer(Box::new(QStarcoder2::try_from(model_config)?))
                }
                GGUFArchitecture::Qwen2 => {
                    ModelVariant::Transformer(Box::new(QQwen::try_from(model_config)?))
                }
                GGUFArchitecture::Qwen3 => {
                    ModelVariant::Transformer(Box::new(QQwen3::try_from(model_config)?))
                }
                GGUFArchitecture::Qwen3MoE => {
                    ModelVariant::Transformer(Box::new(QQwen3MoE::try_from(model_config)?))
                }
                a => bail!("Unsupported architecture `{a:?}` for GGUF"),
            },
            ModelKind::GgufAdapter { adapter, .. } => match arch {
                GGUFArchitecture::Llama => {
                    ModelVariant::XLoraLlama(XLoraQLlama::try_from(model_config)?)
                }
                GGUFArchitecture::Phi3 => {
                    ModelVariant::XLoraPhi3(XLoraQPhi3::try_from(model_config)?)
                }
                GGUFArchitecture::Mistral3 => {
                    bail!("Mistral3 adapters are not supported yet")
                }
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            // For pipeline parallelism, adjust num_layers to loaded layers only.
            // This ensures PagedAttention allocates KV cache proportional to this node's layers,
            // not the full model's layers.
            info!(
                "PagedAttention KV cache: layer_range={:?}, total_layers={}, model_config_num_layers={}",
                self.config.layer_range,
                total_layers,
                model_config_metadata.num_layers()
            );
            let paged_attn_model_config: Box<dyn ModelConfigLike> =
                if let Some(ref range) = self.config.layer_range {
                    Box::new(ModelConfigMetadata {
                        max_seq_len: model_config_metadata.max_seq_len(),
                        num_layers: range.len(),
                        hidden_size: model_config_metadata.hidden_size(),
                        num_kv_heads: model_config_metadata.num_kv_heads(),
                        num_attn_heads: model_config_metadata.num_attn_heads(),
                        sliding_window: None,
                        k_head_dim: model_config_metadata.k_head_dim(),
                        v_head_dim: model_config_metadata.v_head_dim(),
                    })
                } else {
                    Box::new(ModelConfigMetadata {
                        max_seq_len: model_config_metadata.max_seq_len(),
                        num_layers: model_config_metadata.num_layers(),
                        hidden_size: model_config_metadata.hidden_size(),
                        num_kv_heads: model_config_metadata.num_kv_heads(),
                        num_attn_heads: model_config_metadata.num_attn_heads(),
                        sliding_window: None,
                        k_head_dim: model_config_metadata.k_head_dim(),
                        v_head_dim: model_config_metadata.v_head_dim(),
                    })
                };
            info!(
                "PagedAttention KV cache: using num_layers={} for cache calculation",
                paged_attn_model_config.num_layers()
            );
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                internal_dtype,
                paged_attn_config.cache_type,
                paged_attn_model_config.as_ref(),
                device,
                &layer_devices,
                silent,
            )?;
            let cache_engine = CacheEngine::new(
                paged_attn_model_config.as_ref(),
                &cache_config,
                internal_dtype,
                device,
                layer_devices,
            )?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            gguf_chat_template,
        );

        let llg_factory = build_llg_factory(tokenizer.clone())?;

        // Create cache for pipeline (models are stateless, pipeline owns the cache)
        let num_hidden_layers = model.num_layers();
        let max_seq_len = model.max_seq_len();
        let cache = EitherCache::Normal(NormalCache::new(num_hidden_layers, max_seq_len));

        if chat_template.bos_token.is_none() {
            if let Some(v) = bos {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.eos_token.is_none() {
            if let Some(v) = eos {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.unk_token.is_none() {
            if let Some(v) = unk {
                chat_template.unk_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }

        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);

        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self
                .model_id
                .clone()
                .unwrap_or(self.quantized_model_id.clone()),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config,
                cache_engine,
                model_metadata: Some(Arc::new(model_config_metadata)),
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
            }),
            cache,
            mapper: pipeline_mapper,
            hook: None,
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

#[allow(deprecated)]
impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

#[allow(deprecated)]
impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

#[allow(deprecated)]
impl CacheManagerMixin for GGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, false)
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, false)
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        } else {
            NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            );
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

#[allow(deprecated)]
impl MetadataMixin for GGUFPipeline {
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
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[allow(deprecated)]
#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn get_hook(&self) -> Option<&crate::pipeline::HookContainer> {
        self.hook.as_ref()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            context_lens,
            position_ids: _, // NOTE(EricLBuehler): ignore, it is for phi3
            paged_attn_meta,
            flash_meta,
            flash_meta_full,
            request_id,
            inference_step,
        } = *inputs.downcast().expect("Downcast failed.");

        // Set request context on hook BEFORE forward (proper design: context on hook, not threaded through model)
        if let Some(hook) = self.hook.as_ref() {
            hook.set_request_context(request_id);

            use crate::pipeline::text_models_inputs_processor::InferenceStep;
            if let InferenceStep::Prefill { total_prompt_tokens, chunk_start_position, .. } = inference_step {
                hook.call_init_pipeline_request(request_id, total_prompt_tokens, chunk_start_position);
            }
        }

        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            (None, None) => None,
        };
        // Models implementing TransformerModel use the unified forward path.
        // Check BEFORE the match to avoid borrow conflicts.
        if self.model.as_language_model().is_some() {
            // Safe to unwrap: we just checked it's Some
            let model = self.model.as_language_model().unwrap();
            let cache = &mut self.cache.normal().0;
            let hook = self.hook.as_ref();
            return forward_transformer(
                model,
                cache,
                hook,
                &input_ids,
                &seqlen_offsets,
                context_lens,
                paged_attn_meta,
                request_id,
            );
        }

        // Legacy path for XLora models (two-pass inference, own cache)
        let logits = match self.model {
            ModelVariant::XLoraLlama(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            ModelVariant::XLoraPhi3(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            // TransformerModel handled in unified path above
            ModelVariant::Transformer(_) => {
                unreachable!("TransformerModel handled in unified path above")
            }
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
    fn set_hook(&mut self, hook: HookContainer) {
        self.hook = Some(hook);
    }
    fn supports_hooks(&self) -> bool {
        true // Pipeline always supports hooks
    }
}

// TODO
#[allow(deprecated)]
impl AnyMoePipelineMixin for GGUFPipeline {}
