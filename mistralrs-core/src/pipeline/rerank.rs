//! Reranking pipeline for cross-encoder models.
//!
//! This pipeline handles reranking (cross-encoder) models that score
//! query-document relevance. It wraps TEI's CandleBackend in classifier mode.

use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource,
};
use crate::device_map::DeviceMapper;
use crate::paged_attention::PagedAttentionConfig;
use crate::pipeline::chat_template::ChatTemplate;
use crate::pipeline::processing::{BasicProcessor, Processor};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::tei_backend::RerankBackend;
use crate::{DeviceMapSetting, Modalities, Pipeline, SupportedModality, TryIntoDType};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::sync::{Arc, Mutex as StdMutex};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// Inputs for reranking: query-document pairs
pub struct RerankInputs {
    /// The query to rank documents against
    pub query: String,
    /// Documents to rerank
    pub documents: Vec<String>,
    /// Whether to truncate inputs exceeding max length
    pub truncate: bool,
}

/// Reranking pipeline using TEI's cross-encoder backend.
pub struct RerankPipeline {
    /// Wrapped in Mutex because CandleBackend is Send but not Sync
    backend: StdMutex<RerankBackend>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    device: Device,
}

impl RerankPipeline {
    /// Create a new rerank pipeline from a HuggingFace model ID.
    pub fn from_hf(
        model_id: impl Into<String>,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        let model_id = model_id.into();

        let backend = RerankBackend::from_hf(&model_id, revision, token, dtype)
            .context("Failed to load rerank backend")?;

        // Load tokenizer separately for the Pipeline trait
        let tokenizer = Self::load_tokenizer(&model_id, revision, token)?;

        let metadata = Arc::new(GeneralMetadata {
            max_seq_len: 512, // Most BERT models
            llg_factory: None,
            is_xlora: false,
            no_prefix_cache: true,
            num_hidden_layers: 1,
            eos_tok: vec![],
            kind: ModelKind::Normal,
            no_kv_cache: true,
            activation_dtype: DType::F32,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text], // Scores, not embeddings
            },
        });

        Ok(Self {
            backend: StdMutex::new(backend),
            tokenizer: Arc::new(tokenizer),
            model_id,
            metadata,
            device,
        })
    }

    fn load_tokenizer(
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<Tokenizer> {
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

        let mut api_builder = ApiBuilder::new();
        if let Some(token) = token {
            api_builder = api_builder.with_token(Some(token.to_string()));
        }
        let api = api_builder.build().context("Failed to build HF Hub API")?;

        let revision = revision.unwrap_or("main");
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;

        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
    }

    /// Rerank documents against a query.
    pub fn rerank(&self, query: &str, documents: &[String], truncate: bool) -> Result<Vec<f32>> {
        let backend = self
            .backend
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;
        let result = backend.rerank(query, documents, truncate)?;
        Ok(result.scores)
    }

    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl PreProcessingMixin for RerankPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for RerankPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        // TEI backend doesn't support ISQ
        Ok(())
    }
}

impl CacheManagerMixin for RerankPipeline {
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
        unreachable!("RerankPipeline does not use cache")
    }
}

impl MetadataMixin for RerankPipeline {
    fn device(&self) -> Device {
        self.device.clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for RerankPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let RerankInputs {
            query,
            documents,
            truncate,
        } = *inputs.downcast::<RerankInputs>().map_err(|_| {
            candle_core::Error::Msg("Failed to downcast to RerankInputs".to_string())
        })?;

        let backend = self
            .backend
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Lock poisoned: {e}")))?;
        let result = backend
            .rerank(&query, &documents, truncate)
            .map_err(|e| candle_core::Error::Msg(format!("Rerank failed: {e}")))?;

        // Convert scores to tensor
        let scores = Tensor::from_vec(result.scores, (documents.len(),), &self.device)?;

        Ok(ForwardInputsResult::Rerank {
            scores,
            prompt_tokens: result.prompt_tokens,
            total_tokens: result.total_tokens,
        })
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        // Reranking doesn't do causal generation, but we implement for trait compliance
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Rerank
    }
}

impl AnyMoePipelineMixin for RerankPipeline {}

/// Loader for reranking models.
pub struct RerankLoader {
    model_id: String,
    dtype: Option<String>,
}

impl RerankLoader {
    pub fn new(model_id: String, dtype: Option<String>) -> Self {
        Self { model_id, dtype }
    }
}

impl Loader for RerankLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        _dtype: &dyn TryIntoDType,
        device: &Device,
        _silent: bool,
        _mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let token = match &token_source {
            TokenSource::EnvVar(var_name) => std::env::var(var_name).ok(),
            TokenSource::Literal(t) => Some(t.clone()),
            TokenSource::Path(p) => std::fs::read_to_string(p).ok(),
            TokenSource::CacheToken => None, // HF Hub will use cached token
            TokenSource::None => None,
        };

        let pipeline = RerankPipeline::from_hf(
            &self.model_id,
            revision.as_deref(),
            token.as_deref(),
            self.dtype.as_deref(),
            device.clone(),
        )?;

        Ok(Arc::new(Mutex::new(pipeline)))
    }

    fn load_model_from_path(
        &self,
        _paths: &Box<dyn ModelPaths>,
        _dtype: &dyn TryIntoDType,
        device: &Device,
        _silent: bool,
        _mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        // For rerank models, we always load from HF since TEI backend handles paths internally
        let pipeline = RerankPipeline::from_hf(
            &self.model_id,
            None,
            None,
            self.dtype.as_deref(),
            device.clone(),
        )?;

        Ok(Arc::new(Mutex::new(pipeline)))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}
