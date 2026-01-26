//! GGUF Embedding Pipeline - loads quantized GGUF models for embedding generation.
//!
//! This module provides embedding support for GGUF-quantized models, allowing efficient
//! embedding generation without the memory overhead of full-precision weights.

use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, KvCache, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    Modalities, NormalCache, PreProcessingMixin, Processor, SupportedModality,
};
use crate::device_map::DeviceMapper;
use crate::embedding_models::inputs_processor::{EmbeddingProcessor, ModelInputs};
use crate::gguf::{convert_gguf_to_hf_tokenizer, Content, GGUFArchitecture, GgufTokenizerConversion};
use crate::models::quantized_qwen::ModelWeights as QQwen;
use crate::models::quantized_qwen3::ModelWeights as QQwen3;
use crate::models::TransformerModel;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::QuantizationKind;
use crate::utils::model_config as ModelConfig;
use crate::pipeline::ChatTemplate;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::tokens::get_token;
use crate::{
    DeviceMapSetting, LocalModelPaths, PagedAttentionConfig, Pipeline, TokenSource, Topology,
    TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{bail, Result};
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

/// Configuration for GGUF embedding loader.
#[derive(Clone, Default)]
pub struct GGUFEmbeddingSpecificConfig {
    /// Topology configuration for distributed inference.
    pub topology: Option<Topology>,
}

/// Builder for GGUF embedding loader.
#[derive(Default)]
pub struct GGUFEmbeddingLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    config: GGUFEmbeddingSpecificConfig,
}

impl GGUFEmbeddingLoaderBuilder {
    /// Create a new GGUF embedding loader builder.
    ///
    /// # Arguments
    /// * `tok_model_id` - Optional model ID for tokenizer (if different from quantized model)
    /// * `quantized_model_id` - HuggingFace repo or local path containing the GGUF file
    /// * `quantized_filenames` - GGUF filename(s)
    /// * `config` - Loader-specific configuration
    pub fn new(
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFEmbeddingSpecificConfig,
    ) -> Self {
        Self {
            model_id: tok_model_id,
            quantized_model_id,
            quantized_filenames,
            config,
        }
    }

    /// Build the loader.
    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFEmbeddingLoader {
            model_id: self.model_id,
            quantized_model_id: self.quantized_model_id,
            quantized_filenames: self.quantized_filenames,
            config: self.config,
            kind: ModelKind::GgufQuantized {
                quant: QuantizationKind::Gguf,
            },
        })
    }
}

/// Loader for GGUF embedding models.
pub struct GGUFEmbeddingLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    config: GGUFEmbeddingSpecificConfig,
    kind: ModelKind,
}

impl GGUFEmbeddingLoader {
    /// Create a new GGUF embedding loader.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFEmbeddingSpecificConfig,
    ) -> Self {
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            config,
            kind: ModelKind::GgufQuantized {
                quant: QuantizationKind::Gguf,
            },
        }
    }
}

impl Loader for GGUFEmbeddingLoader {
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
        let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();

        // Build HF API
        let api = ApiBuilder::from_cache(cache)
            .with_progress(!silent)
            .with_token(get_token(&token_source)?)
            .build()?;

        let revision = revision.unwrap_or_else(|| "main".to_string());

        // Get GGUF weight files
        let quantized_api = api.repo(Repo::with_revision(
            self.quantized_model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));

        let mut weight_filenames = Vec::new();
        for filename in &self.quantized_filenames {
            weight_filenames.push(quantized_api.get(filename)?);
        }

        // Get tokenizer if model_id specified
        let tokenizer_filename = if let Some(ref tok_model_id) = self.model_id {
            let tok_api = api.repo(Repo::with_revision(
                tok_model_id.clone(),
                RepoType::Model,
                revision,
            ));
            tok_api.get("tokenizer.json").ok()
        } else {
            None
        };

        // Create paths struct
        let paths: Box<dyn ModelPaths> = Box::new(LocalModelPaths::new(
            tokenizer_filename.unwrap_or_default(),
            PathBuf::new(),                             // No separate config file for GGUF
            PathBuf::new(),                             // No template file
            weight_filenames,
            crate::pipeline::paths::AdapterPaths::None, // No adapter paths
            None,                                       // No gen_conf
            None,                                       // No preprocessor_config
            None,                                       // No processor_config
            None,                                       // No chat_template_json
        ));

        self.load_model_from_path(&paths, dtype, device, silent, mapper, in_situ_quant, paged_attn_config)
    }

    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);

        // Load GGUF content from paths
        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers_ref = readers.iter_mut().collect::<Vec<_>>();

        let content = Content::from_readers(&mut readers_ref)?;
        let arch = content.arch();

        info!("Loading GGUF embedding model with architecture: {:?}", arch);

        // Get number of layers for device mapping
        let num_layers = content.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        // Handle auto device mapping - for embeddings just use dummy (all on primary device)
        if matches!(mapper, DeviceMapSetting::Auto(_)) {
            mapper = DeviceMapSetting::dummy();
        }

        let pipeline_mapper =
            mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;
        let model_mapper =
            mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;

        // Get tokenizer
        let GgufTokenizerConversion {
            tokenizer,
            bos: _,
            eos: _,
            unk: _,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&content)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Get internal dtype
        let internal_dtype = pipeline_mapper.get_min_dtype(dtype)?;

        // Extract max_seq_len from metadata before content is consumed
        let max_seq_len = content
            .get_metadata()
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(8192) as usize;

        // Create model config for loading
        let model_config = {
            let quant = ModelConfig::ParamsGGUF(
                content,
                (device, model_mapper).into(),
                AttentionImplementation::Eager, // No paged attention for embeddings
                internal_dtype,
                None, // No layer range (not pipeline parallel)
            );
            ModelConfig::ModelParams::new(quant, None)
        };

        // Load the appropriate model based on architecture
        let model: GGUFEmbeddingModel = match arch {
            GGUFArchitecture::Qwen2 => {
                info!("Loading Qwen2 architecture for embeddings");
                GGUFEmbeddingModel::Qwen(QQwen::try_from(model_config)?)
            }
            GGUFArchitecture::Qwen3 => {
                info!("Loading Qwen3 architecture for embeddings");
                GGUFEmbeddingModel::Qwen3(QQwen3::try_from(model_config)?)
            }
            a => bail!(
                "Unsupported architecture `{a:?}` for GGUF embedding. \
                Supported: Qwen2, Qwen3"
            ),
        };

        // Create metadata for embedding pipeline
        let metadata = Arc::new(GeneralMetadata {
            max_seq_len,
            llg_factory: None,
            no_kv_cache: true,
            no_prefix_cache: true,
            num_hidden_layers: num_layers,
            eos_tok: vec![],
            kind: ModelKind::GgufQuantized {
                quant: QuantizationKind::Gguf,
            },
            is_xlora: false,
            activation_dtype: internal_dtype,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Embedding],
            },
        });

        // Create processor
        let processor = Arc::new(EmbeddingProcessor {
            has_causal_attention: false, // Bidirectional attention for embeddings
        });

        // Create cache for pipeline (models are stateless)
        let num_layers = model.num_layers();
        let max_seq_len = model.max_seq_len();
        let cache = EitherCache::Normal(NormalCache::new(num_layers, max_seq_len));

        let pipeline = GGUFEmbeddingPipeline {
            model,
            tokenizer: Arc::new(tokenizer),
            model_id: self.quantized_model_id.clone(),
            metadata,
            mapper: pipeline_mapper,
            processor,
            device: device.clone(),
            cache,
        };

        Ok(Arc::new(Mutex::new(pipeline)))
    }

    fn get_id(&self) -> String {
        self.quantized_model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

/// Enum of supported GGUF embedding model architectures.
enum GGUFEmbeddingModel {
    Qwen(QQwen),
    Qwen3(QQwen3),
}

impl GGUFEmbeddingModel {
    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize {
        match self {
            GGUFEmbeddingModel::Qwen(model) => model.num_layers(),
            GGUFEmbeddingModel::Qwen3(model) => model.num_layers(),
        }
    }

    /// Maximum sequence length supported by this model.
    fn max_seq_len(&self) -> usize {
        match self {
            GGUFEmbeddingModel::Qwen(model) => model.max_seq_len,
            GGUFEmbeddingModel::Qwen3(model) => model.max_seq_len,
        }
    }

    /// Forward pass to generate embeddings (hidden states before LM head).
    fn forward_hidden_states(
        &self,
        input_ids: &Tensor,
        cache: &mut [KvCache],
    ) -> Result<Tensor, candle_core::Error> {
        let (batch_size, _seq_len) = input_ids.dims2()?;
        let start_offsets = vec![0usize; batch_size];

        match self {
            GGUFEmbeddingModel::Qwen(model) => {
                model.forward_hidden_states(input_ids, &start_offsets, cache)
            }
            GGUFEmbeddingModel::Qwen3(model) => {
                model.forward_hidden_states(input_ids, &start_offsets, cache)
            }
        }
    }

    #[allow(dead_code)]
    fn device(&self) -> &Device {
        match self {
            GGUFEmbeddingModel::Qwen(model) => &model.device,
            GGUFEmbeddingModel::Qwen3(model) => &model.device,
        }
    }
}

/// GGUF Embedding Pipeline - generates embeddings from GGUF-quantized models.
pub struct GGUFEmbeddingPipeline {
    model: GGUFEmbeddingModel,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    processor: Arc<dyn Processor + Send + Sync>,
    device: Device,
    /// KV cache owned by the pipeline (models are stateless)
    cache: EitherCache,
}

#[async_trait]
impl Pipeline for GGUFEmbeddingPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            flash_meta: _,
        } = *inputs
            .downcast::<ModelInputs>()
            .map_err(|_| candle_core::Error::Msg("Invalid input type for embedding pipeline".to_string()))?;

        // Get hidden states from model
        let cache = &mut self.cache.normal().0;
        let hidden_states = self.model.forward_hidden_states(&input_ids, cache)?;

        // Apply last-token pooling (take the last token's hidden state for each sequence)
        // This is appropriate for Qwen3-Embedding models
        let (_batch_size, seq_len, _hidden_dim) = hidden_states.dims3()?;
        let embeddings = hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        Ok(ForwardInputsResult::Embeddings { embeddings })
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        Ok(())
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Embedding
    }
}

impl MetadataMixin for GGUFEmbeddingPipeline {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }

    fn name(&self) -> String {
        self.model_id.clone()
    }

    fn reset_non_granular_state(&self) {}

    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }

    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(self.mapper.as_ref())
    }
}

impl PreProcessingMixin for GGUFEmbeddingPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }

    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }

    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl CacheManagerMixin for GGUFEmbeddingPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

impl IsqPipelineMixin for GGUFEmbeddingPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        Ok(())
    }
}

impl AnyMoePipelineMixin for GGUFEmbeddingPipeline {}
