//! Text Embeddings Inference (TEI) backend integration
//!
//! This module provides embedding and reranking capabilities using HuggingFace's
//! Text Embeddings Inference (TEI) Candle backend. It supports BERT-family models
//! for both dense embeddings and cross-encoder reranking.
//!
//! # Supported Model Types
//!
//! - **Embeddings**: BERT, RoBERTa, DistilBERT, XLM-RoBERTa, GTE, ModernBERT, NomicBERT, etc.
//! - **Reranking**: Any cross-encoder model (BERT with classification head)
//!
//! # Example
//!
//! ```ignore
//! use mistralrs_core::tei_backend::{TeiBackend, TeiConfig};
//!
//! let config = TeiConfig {
//!     model_path: PathBuf::from("/path/to/model"),
//!     dtype: "float16".to_string(),
//!     use_classifier: false, // true for reranking
//! };
//!
//! let backend = TeiBackend::new(config)?;
//! let embeddings = backend.embed(texts)?;
//! ```

use std::path::PathBuf;

use text_embeddings_backend_core::{BackendError, Batch, Pool};

#[cfg(feature = "tei-backend")]
use text_embeddings_backend_candle::CandleBackend;

/// Configuration for the TEI backend
#[derive(Debug, Clone)]
pub struct TeiConfig {
    /// Path to the model directory (containing config.json, model.safetensors, etc.)
    pub model_path: PathBuf,
    /// Data type: "float32" or "float16"
    pub dtype: String,
    /// Pooling strategy for embeddings
    pub pool: TeiPool,
    /// Whether this is a classifier (reranking) model
    pub is_classifier: bool,
    /// Optional paths to dense layers (for sentence-transformers models)
    pub dense_paths: Option<Vec<String>>,
}

/// Pooling strategies for embeddings
#[derive(Debug, Clone, Default)]
pub enum TeiPool {
    /// Use CLS token embedding
    Cls,
    /// Mean pooling over all tokens
    #[default]
    Mean,
    /// SPLADE sparse embeddings
    Splade,
    /// Use last token embedding (for decoder models)
    LastToken,
}

impl From<TeiPool> for Pool {
    fn from(pool: TeiPool) -> Self {
        match pool {
            TeiPool::Cls => Pool::Cls,
            TeiPool::Mean => Pool::Mean,
            TeiPool::Splade => Pool::Splade,
            TeiPool::LastToken => Pool::LastToken,
        }
    }
}

/// Wrapper around TEI's CandleBackend for embedding and reranking
#[cfg(feature = "tei-backend")]
pub struct TeiBackend {
    backend: CandleBackend,
    is_classifier: bool,
}

#[cfg(feature = "tei-backend")]
impl TeiBackend {
    /// Create a new TEI backend from configuration
    pub fn new(config: TeiConfig) -> Result<Self, BackendError> {
        let model_type = if config.is_classifier {
            ModelType::Classifier
        } else {
            ModelType::Embedding(config.pool.into())
        };

        let backend = CandleBackend::new(
            &config.model_path,
            config.dtype,
            model_type,
            config.dense_paths,
        )?;

        Ok(Self {
            backend,
            is_classifier: config.is_classifier,
        })
    }

    /// Check if the backend is healthy
    pub fn health(&self) -> Result<(), BackendError> {
        self.backend.health()
    }

    /// Get maximum batch size (if limited)
    pub fn max_batch_size(&self) -> Option<usize> {
        self.backend.max_batch_size()
    }

    /// Whether the model uses padding
    pub fn is_padded(&self) -> bool {
        self.backend.is_padded()
    }

    /// Compute embeddings for a batch of tokenized inputs
    ///
    /// # Arguments
    /// * `batch` - Tokenized inputs in TEI Batch format
    ///
    /// # Returns
    /// Map of index -> embedding vectors
    pub fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        if self.is_classifier {
            return Err(BackendError::Inference(
                "Cannot embed with a classifier model".to_string(),
            ));
        }
        self.backend.embed(batch)
    }

    /// Compute predictions (reranking scores) for a batch of tokenized inputs
    ///
    /// # Arguments
    /// * `batch` - Tokenized query-document pairs in TEI Batch format
    ///
    /// # Returns
    /// Map of index -> prediction scores
    pub fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        if !self.is_classifier {
            return Err(BackendError::Inference(
                "Cannot predict with an embedding model".to_string(),
            ));
        }
        self.backend.predict(batch)
    }
}

/// Helper to create a TEI Batch from tokenized text
///
/// # Arguments
/// * `input_ids` - Token IDs for all sequences, concatenated
/// * `token_type_ids` - Token type IDs (0 for first segment, 1 for second in pair tasks)
/// * `position_ids` - Position IDs for each token
/// * `cumulative_seq_lengths` - Cumulative lengths: [0, len1, len1+len2, ...]
/// * `pooled_indices` - Which sequences need pooled embeddings
/// * `raw_indices` - Which sequences need per-token embeddings
pub fn create_batch(
    input_ids: Vec<u32>,
    token_type_ids: Vec<u32>,
    position_ids: Vec<u32>,
    cumulative_seq_lengths: Vec<u32>,
    pooled_indices: Vec<u32>,
    raw_indices: Vec<u32>,
) -> Batch {
    let max_length = cumulative_seq_lengths
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0);

    Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length,
        pooled_indices,
        raw_indices,
    }
}

/// Stub implementation when tei-backend feature is not enabled
#[cfg(not(feature = "tei-backend"))]
pub struct TeiBackend;

#[cfg(not(feature = "tei-backend"))]
impl TeiBackend {
    pub fn new(_config: TeiConfig) -> Result<Self, BackendError> {
        Err(BackendError::NoBackend)
    }
}

// Re-export core types for convenience
pub use text_embeddings_backend_core::{Embedding as TeiEmbedding, BackendError as TeiBackendError};

/// Specifies whether to load an embedding or reranking model
#[derive(Debug, Clone, Copy, Default)]
pub enum TeiModelKind {
    /// Embedding model (default)
    #[default]
    Embedding,
    /// Reranking/classifier model
    Reranker,
}

/// Configuration for loading TEI models
#[derive(Debug, Clone, Default)]
pub struct TeiSpecificConfig {
    /// Data type: "float32" or "float16"
    pub dtype: Option<String>,
    /// Pooling strategy for embeddings
    pub pool: Option<TeiPool>,
    /// Dense layer paths for sentence-transformers models
    pub dense_paths: Option<Vec<String>>,
}

/// Builder for loading TEI models from HuggingFace Hub
///
/// # Example
/// ```ignore
/// let backend = TeiLoaderBuilder::new(
///     TeiSpecificConfig::default(),
///     "BAAI/bge-small-en-v1.5".to_string(),
///     TeiModelKind::Embedding,
/// )
/// .with_revision("main")
/// .build()?;
/// ```
#[cfg(feature = "tei-backend")]
#[derive(Default)]
pub struct TeiLoaderBuilder {
    model_id: Option<String>,
    config: TeiSpecificConfig,
    kind: TeiModelKind,
    revision: Option<String>,
    token: Option<String>,
}

#[cfg(feature = "tei-backend")]
impl TeiLoaderBuilder {
    /// Create a new TEI loader builder
    ///
    /// # Arguments
    /// * `config` - TEI-specific configuration
    /// * `model_id` - HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
    /// * `kind` - Whether to load as embedding or reranking model
    pub fn new(config: TeiSpecificConfig, model_id: String, kind: TeiModelKind) -> Self {
        Self {
            model_id: Some(model_id),
            config,
            kind,
            revision: None,
            token: None,
        }
    }

    /// Set the model revision (branch, tag, or commit hash)
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set the HuggingFace token for private models
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Build and load the TEI backend
    pub fn build(self) -> anyhow::Result<TeiBackend> {
        use anyhow::Context;
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

        let model_id = self
            .model_id
            .ok_or_else(|| anyhow::anyhow!("model_id is required"))?;

        // Build the HF Hub API
        let mut api_builder = ApiBuilder::new();
        if let Some(token) = &self.token {
            api_builder = api_builder.with_token(Some(token.clone()));
        }
        let api = api_builder
            .build()
            .context("Failed to build HuggingFace Hub API")?;

        // Get the repository
        let revision = self.revision.as_deref().unwrap_or("main");
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision.to_string(),
        ));

        // Download required files
        let model_path = repo
            .get("config.json")
            .context("Failed to download config.json")?
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?
            .to_path_buf();

        // Try to download model weights (safetensors preferred)
        let _ = repo
            .get("model.safetensors")
            .or_else(|_| {
                // Try sharded safetensors
                repo.get("model.safetensors.index.json")
                    .or_else(|_| repo.get("pytorch_model.bin"))
            })
            .context("Failed to download model weights")?;

        // Download tokenizer if exists (ignore errors)
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        let _ = repo.get("special_tokens_map.json");

        // Determine pooling and classifier settings based on kind
        let (is_classifier, default_pool) = match self.kind {
            TeiModelKind::Embedding => (false, TeiPool::Mean),
            TeiModelKind::Reranker => (true, TeiPool::Cls),
        };

        // Create the config and load the backend
        let config = TeiConfig {
            model_path,
            dtype: self.config.dtype.unwrap_or_else(|| "float32".to_string()),
            pool: self.config.pool.unwrap_or(default_pool),
            is_classifier,
            dense_paths: self.config.dense_paths,
        };

        TeiBackend::new(config).map_err(|e| anyhow::anyhow!("TEI backend error: {e}"))
    }
}

// ============================================================================
// High-level Reranking Backend
// ============================================================================

/// Result of a reranking operation
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Relevance scores for each document (in input order)
    pub scores: Vec<f32>,
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Total tokens processed
    pub total_tokens: usize,
}

/// High-level reranking backend with tokenization support
///
/// This struct wraps TeiBackend and provides a simple API for reranking
/// that handles tokenization internally.
#[cfg(feature = "tei-backend")]
pub struct RerankBackend {
    backend: TeiBackend,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
    model_id: String,
}

#[cfg(feature = "tei-backend")]
impl RerankBackend {
    /// Maximum sequence length for most BERT models
    const DEFAULT_MAX_LENGTH: usize = 512;

    /// Load a reranking model from HuggingFace Hub
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "BAAI/bge-reranker-base")
    /// * `revision` - Optional revision (branch, tag, or commit)
    /// * `token` - Optional HuggingFace token for private models
    /// * `dtype` - Data type ("float32" or "float16")
    pub fn from_hf(
        model_id: impl Into<String>,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<&str>,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

        let model_id = model_id.into();

        // Build the HF Hub API
        let mut api_builder = ApiBuilder::new();
        if let Some(token) = token {
            api_builder = api_builder.with_token(Some(token.to_string()));
        }
        let api = api_builder
            .build()
            .context("Failed to build HuggingFace Hub API")?;

        // Get the repository
        let revision = revision.unwrap_or("main");
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision.to_string(),
        ));

        // Download required files
        let model_path = repo
            .get("config.json")
            .context("Failed to download config.json")?
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?
            .to_path_buf();

        // Try to download model weights
        let _ = repo
            .get("model.safetensors")
            .or_else(|_| {
                repo.get("model.safetensors.index.json")
                    .or_else(|_| repo.get("pytorch_model.bin"))
            })
            .context("Failed to download model weights")?;

        // Download and load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Try to get max_length from tokenizer_config.json
        let max_length = repo
            .get("tokenizer_config.json")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
            .and_then(|config| config.get("model_max_length")?.as_u64())
            .map(|v| v as usize)
            .unwrap_or(Self::DEFAULT_MAX_LENGTH);

        // Create the TEI backend (classifier mode for reranking)
        let config = TeiConfig {
            model_path,
            dtype: dtype.unwrap_or("float32").to_string(),
            pool: TeiPool::Cls,
            is_classifier: true,
            dense_paths: None,
        };

        let backend =
            TeiBackend::new(config).map_err(|e| anyhow::anyhow!("TEI backend error: {e}"))?;

        Ok(Self {
            backend,
            tokenizer,
            max_length,
            model_id,
        })
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Rerank documents against a query
    ///
    /// # Arguments
    /// * `query` - The query to rank documents against
    /// * `documents` - Documents to rerank
    /// * `truncate` - Whether to truncate inputs that exceed max length
    ///
    /// # Returns
    /// Relevance scores for each document (in input order)
    pub fn rerank(
        &self,
        query: &str,
        documents: &[String],
        truncate: bool,
    ) -> anyhow::Result<RerankResult> {
        if documents.is_empty() {
            return Ok(RerankResult {
                scores: vec![],
                prompt_tokens: 0,
                total_tokens: 0,
            });
        }

        // Tokenize all query-document pairs
        let mut all_input_ids: Vec<u32> = Vec::new();
        let mut all_token_type_ids: Vec<u32> = Vec::new();
        let mut all_position_ids: Vec<u32> = Vec::new();
        let mut cumulative_seq_lengths: Vec<u32> = vec![0];
        let mut total_tokens: usize = 0;

        for doc in documents {
            let encoding = self
                .tokenizer
                .encode((query, doc.as_str()), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

            let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
            let mut token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

            // Truncate if needed
            if input_ids.len() > self.max_length {
                if truncate {
                    input_ids.truncate(self.max_length);
                    token_type_ids.truncate(self.max_length);
                } else {
                    anyhow::bail!(
                        "Input length {} exceeds max length {}. Set truncate=true to allow truncation.",
                        input_ids.len(),
                        self.max_length
                    );
                }
            }

            let seq_len = input_ids.len();
            total_tokens += seq_len;

            // Position IDs: 0, 1, 2, ... for this sequence
            let position_ids: Vec<u32> = (0..seq_len as u32).collect();

            all_input_ids.extend(input_ids);
            all_token_type_ids.extend(token_type_ids);
            all_position_ids.extend(position_ids);
            cumulative_seq_lengths.push(all_input_ids.len() as u32);
        }

        // All sequences need pooled predictions
        let pooled_indices: Vec<u32> = (0..documents.len() as u32).collect();

        let batch = create_batch(
            all_input_ids,
            all_token_type_ids,
            all_position_ids,
            cumulative_seq_lengths,
            pooled_indices,
            vec![], // No raw indices needed for reranking
        );

        // Run prediction
        let predictions = self
            .backend
            .predict(batch)
            .map_err(|e| anyhow::anyhow!("Prediction failed: {e}"))?;

        // Extract scores in order
        // Predictions is IntMap<usize, Vec<f32>> where Vec<f32> contains class scores
        let mut scores = vec![0.0f32; documents.len()];
        for (idx, prediction_scores) in predictions {
            if idx < scores.len() {
                // For binary classification, use the positive class score
                // prediction_scores contains [negative_score, positive_score]
                scores[idx] = if prediction_scores.len() > 1 {
                    prediction_scores[1]
                } else {
                    prediction_scores[0]
                };
            }
        }

        Ok(RerankResult {
            scores,
            prompt_tokens: total_tokens,
            total_tokens,
        })
    }

    /// Check if the backend is healthy
    pub fn health(&self) -> Result<(), BackendError> {
        self.backend.health()
    }
}

/// Stub implementation when tei-backend feature is not enabled
#[cfg(not(feature = "tei-backend"))]
pub struct RerankBackend;

#[cfg(not(feature = "tei-backend"))]
impl RerankBackend {
    pub fn from_hf(
        _model_id: impl Into<String>,
        _revision: Option<&str>,
        _token: Option<&str>,
        _dtype: Option<&str>,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("TEI backend feature is not enabled. Compile with --features tei-backend")
    }

    pub fn model_id(&self) -> &str {
        ""
    }

    pub fn rerank(
        &self,
        _query: &str,
        _documents: &[String],
        _truncate: bool,
    ) -> anyhow::Result<RerankResult> {
        anyhow::bail!("TEI backend feature is not enabled")
    }

    pub fn health(&self) -> Result<(), BackendError> {
        Err(BackendError::NoBackend)
    }
}
