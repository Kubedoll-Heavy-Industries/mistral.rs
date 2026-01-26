//! GGUF format metadata extraction.
//!
//! This module provides a clean interface for opening GGUF files and extracting
//! their metadata, tokenizer, and chat template - without constructing models.
//!
//! # Model Construction
//!
//! Model construction is handled separately via `FromGGUF` trait. Each model
//! parses its own config from GGUF metadata using `ContentMetadata`:
//!
//! ```ignore
//! // In model implementation (e.g., quantized_qwen3.rs)
//! let metadata = ContentMetadata {
//!     path_prefix: "qwen3",
//!     metadata: content.get_metadata(),
//! };
//! let props = PropsGGUF::try_from(metadata)?;
//! ```

// GGUF metadata uses u32 for counts; conversion to usize is safe on 64-bit
#![allow(clippy::cast_possible_truncation)]

use std::any::Any;
use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::device_map::{self, DeviceMapSetting, DeviceMapper};
use crate::gguf::{convert_gguf_to_hf_tokenizer, get_gguf_chat_template, Content, GGUFArchitecture};
use crate::utils::gguf_metadata::GgufDeviceMapLoaderInner;
use crate::{AutoDeviceMapParams, DeviceMappedModelLoader, NonMappedSubModel};
use crate::models::LanguageModel;
use crate::paged_attention::{
    calculate_cache_config, AttentionImplementation, CacheEngine, ModelConfigLike,
    ModelConfigMetadata, PagedAttentionConfig,
};
use crate::pipeline::chat_template::calculate_eos_tokens;
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::TokenSource;
use crate::utils::gguf_metadata::ContentConfig;
use crate::utils::model_config::FromGGUF;
use mistralrs_quant::IsqType;
use regex::Regex;

// Type aliases for models (used in architecture dispatch)
use crate::models::llama::LlamaModel;
use crate::models::mixtral::Mixtral;
use crate::models::quantized_mistral3::ModelWeights as QMistral3;
use crate::models::quantized_phi2::ModelWeights as QPhi2;
use crate::models::quantized_phi3::ModelWeights as QPhi3;
use crate::models::quantized_qwen::ModelWeights as QQwen;
use crate::models::quantized_qwen3::ModelWeights as QQwen3;
use crate::models::quantized_qwen3_moe::ModelWeights as QQwen3MoE;
use crate::models::quantized_starcoder2::ModelWeights as QStarcoder2;

/// Format-specific metadata.
///
/// Use downcasting to access format-specific fields:
/// ```ignore
/// if let Some(gguf) = metadata.as_any().downcast_ref::<GgufMetadata>() {
///     println!("Architecture: {:?}", gguf.architecture);
///     println!("RoPE freq base: {:?}", gguf.rope_freq_base);
/// }
/// ```
#[allow(dead_code)] // Future: used by unified loader architecture
pub trait LoaderMetadata: Send + Sync {
    /// Convert to Any for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// Common interface for model loaders.
///
/// Loaders open model files and extract their components. Different formats
/// (GGUF, Safetensors) implement this trait to provide a unified interface
/// for querying model properties.
///
/// # Format Differences
///
/// | Component | GGUF | Safetensors |
/// |-----------|------|-------------|
/// | Weights | Embedded | `.safetensors` files |
/// | Tokenizer | Embedded | `tokenizer.json` |
/// | Chat template | Embedded | `tokenizer_config.json` |
/// | Config | GGUF metadata | `config.json` |
///
/// # Usage
///
/// ```ignore
/// fn estimate_memory(loader: &dyn MetadataLoader) -> usize {
///     let layers = loader.num_layers();
///     let hidden = loader.hidden_size();
///     // ... compute memory estimate
/// }
/// ```
#[allow(dead_code)] // Future: used by unified loader architecture
pub trait MetadataLoader: Send + Sync {
    /// Architecture as a HuggingFace-style string (e.g., "LlamaForCausalLM").
    fn architecture(&self) -> Result<String>;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Hidden dimension size.
    fn hidden_size(&self) -> usize;

    /// Maximum sequence length (context window).
    fn max_seq_len(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> usize;

    /// Number of key-value heads (for GQA/MQA).
    fn num_kv_heads(&self) -> usize;

    /// Get the tokenizer.
    fn tokenizer(&self) -> Result<Arc<Tokenizer>>;

    /// Get the raw chat template string (Jinja2 format), if available.
    fn chat_template_string(&self) -> Result<Option<String>>;

    /// Format-specific metadata via downcasting.
    ///
    /// ```ignore
    /// if let Some(gguf) = loader.format_metadata().as_any().downcast_ref::<GgufMetadata>() {
    ///     println!("RoPE freq base: {:?}", gguf.rope_freq_base);
    /// }
    /// ```
    fn format_metadata(&self) -> &dyn LoaderMetadata;
}

/// GGUF-specific metadata.
///
/// Contains format-specific fields that can be accessed via downcasting:
/// ```ignore
/// if let Some(gguf) = loader.format_metadata().as_any().downcast_ref::<GgufMetadata>() {
///     println!("Architecture: {:?}", gguf.architecture);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// Detected architecture (Llama, Qwen, Phi, etc.)
    pub architecture: GGUFArchitecture,

    /// Model name from metadata
    pub model_name: Option<String>,

    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,

    /// RoPE scaling type
    pub rope_scaling_type: Option<String>,

    /// Context length from metadata
    pub context_length: Option<usize>,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads
    pub num_kv_heads: usize,

    /// Hidden dimension
    pub hidden_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of experts (0 for dense models, >1 for MoE)
    pub expert_count: usize,
}

impl LoaderMetadata for GgufMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// GGUF file loader for metadata extraction.
///
/// Opens GGUF files and provides access to their components:
/// - Architecture detection
/// - Tokenizer (embedded)
/// - Chat template (embedded)
/// - Metadata queries (num_layers, hidden_size, etc.)
///
/// # Example
///
/// ```ignore
/// let loader = GgufLoader::open(&["model.gguf"])?;
/// let arch = loader.architecture();
/// let tokenizer = loader.tokenizer()?;
///
/// // Model construction is separate - use FromGGUF trait
/// // Architecture dispatch uses GGUFArchitecture enum
/// ```
#[derive(Debug)]
pub struct GgufLoader {
    /// Paths to GGUF files (may be sharded)
    paths: Vec<PathBuf>,

    /// Cached metadata (extracted during construction)
    metadata: GgufMetadata,
}

impl GgufLoader {
    /// Open GGUF file(s) and extract metadata.
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to .gguf file(s). For sharded models, provide all shards.
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<Self> {
        if paths.is_empty() {
            bail!("No GGUF files provided");
        }

        let paths: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();

        // Parse metadata from files
        let metadata = Self::extract_metadata(&paths)?;

        Ok(Self { paths, metadata })
    }

    /// Extract metadata from GGUF files without keeping them open.
    fn extract_metadata(paths: &[PathBuf]) -> Result<GgufMetadata> {
        let mut readers: Vec<BufReader<File>> = paths
            .iter()
            .map(|p| Ok(BufReader::new(File::open(p)?)))
            .collect::<Result<Vec<_>>>()?;

        let mut reader_refs: Vec<&mut BufReader<File>> = readers.iter_mut().collect();
        let content = Content::from_readers(&mut reader_refs)?;

        let arch = content.arch();
        let gguf_metadata = content.get_metadata();

        // Extract metadata
        let arch_str = arch.to_string().to_lowercase();
        let block_count = gguf_metadata
            .get(&format!("{arch_str}.block_count"))
            .map(|v| v.to_u32().unwrap_or(0) as usize)
            .unwrap_or(0);

        let head_count = gguf_metadata
            .get(&format!("{arch_str}.attention.head_count"))
            .map(|v| v.to_u32().unwrap_or(0) as usize)
            .unwrap_or(0);

        let head_count_kv = gguf_metadata
            .get(&format!("{arch_str}.attention.head_count_kv"))
            .map(|v| v.to_u32().unwrap_or(head_count as u32) as usize)
            .unwrap_or(head_count);

        let embedding_length = gguf_metadata
            .get(&format!("{arch_str}.embedding_length"))
            .map(|v| v.to_u32().unwrap_or(0) as usize)
            .unwrap_or(0);

        let vocab_size = gguf_metadata
            .get(&format!("{arch_str}.vocab_size"))
            .or_else(|| gguf_metadata.get("tokenizer.ggml.tokens"))
            .map(|v| match v {
                Value::Array(arr) => arr.len(),
                _ => v.to_u32().unwrap_or(0) as usize,
            })
            .unwrap_or(0);

        let context_length = gguf_metadata
            .get(&format!("{arch_str}.context_length"))
            .map(|v| v.to_u32().unwrap_or(4096) as usize);

        let rope_freq_base = gguf_metadata
            .get(&format!("{arch_str}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok());

        let model_name = gguf_metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string());

        let expert_count = gguf_metadata
            .get(&format!("{arch_str}.expert_count"))
            .map(|v| v.to_u32().unwrap_or(0) as usize)
            .unwrap_or(0);

        Ok(GgufMetadata {
            architecture: arch,
            model_name,
            rope_freq_base,
            rope_scaling_type: None,
            context_length,
            num_attention_heads: head_count,
            num_kv_heads: head_count_kv,
            hidden_size: embedding_length,
            num_layers: block_count,
            vocab_size,
            expert_count,
        })
    }

    /// Get the detected architecture enum.
    pub fn gguf_architecture(&self) -> GGUFArchitecture {
        self.metadata.architecture
    }

    /// Get the architecture as a HuggingFace-style string.
    pub fn architecture_string(&self) -> Result<String> {
        let name = match self.metadata.architecture {
            GGUFArchitecture::Llama => "LlamaForCausalLM",
            GGUFArchitecture::Phi2 => "PhiForCausalLM",
            GGUFArchitecture::Phi3 => "Phi3ForCausalLM",
            GGUFArchitecture::Qwen2 => "Qwen2ForCausalLM",
            GGUFArchitecture::Qwen3 => "Qwen3ForCausalLM",
            GGUFArchitecture::Qwen3MoE => "Qwen3MoeForCausalLM",
            GGUFArchitecture::Starcoder2 => "Starcoder2ForCausalLM",
            GGUFArchitecture::Mistral3 => "MistralForCausalLM",
            _ => {
                return Err(anyhow!(
                    "Unsupported GGUF architecture: {:?}",
                    self.metadata.architecture
                ))
            }
        };
        Ok(name.to_string())
    }

    /// Get the loader metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Get paths to the GGUF files.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    // ========================================================================
    // Memory Estimation
    // ========================================================================

    /// Get per-layer weight sizes in bytes.
    ///
    /// Reads actual tensor sizes from GGUF metadata, accounting for:
    /// - Quantization block overhead
    /// - Architecture-specific components (biases, QK norm, etc.)
    /// - MoE expert tensors (all experts loaded together)
    ///
    /// # Returns
    ///
    /// Vector of layer sizes in bytes, one per transformer layer. All layers
    /// are assumed to have the same size (uses layer 0 as reference).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loader = GgufLoader::open(&["model.gguf"])?;
    /// let layer_sizes = loader.layer_sizes_in_bytes(DType::F16)?;
    /// let total_layers_bytes: usize = layer_sizes.iter().sum();
    /// ```
    pub fn layer_sizes_in_bytes(&self, dtype: DType) -> Result<Vec<usize>> {
        // Open files as File directly (not BufReader) for GgufDeviceMapLoaderInner
        let mut files: Vec<File> = self
            .paths
            .iter()
            .map(|p| File::open(p).map_err(|e| anyhow!("Failed to open {}: {}", p.display(), e)))
            .collect::<Result<Vec<_>>>()?;

        let mut file_refs: Vec<&mut File> = files.iter_mut().collect();
        let content = Content::from_readers(&mut file_refs)?;

        let inner = GgufDeviceMapLoaderInner {
            model: &content,
            arch: self.metadata.architecture,
        };

        inner.layer_sizes_in_bytes("", dtype, 1, None)
    }

    /// Get non-mapped (embedding layer) size in bytes.
    ///
    /// Returns the size of tensors that are NOT mapped across devices:
    /// - Token embeddings (dequantized to F32)
    /// - Output norm
    /// - LM head (output projection)
    ///
    /// These tensors are typically kept on the first device.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loader = GgufLoader::open(&["model.gguf"])?;
    /// let embedding_bytes = loader.non_mapped_size_in_bytes(DType::F16)?;
    /// ```
    pub fn non_mapped_size_in_bytes(&self, dtype: DType) -> Result<usize> {
        let mut files: Vec<File> = self
            .paths
            .iter()
            .map(|p| File::open(p).map_err(|e| anyhow!("Failed to open {}: {}", p.display(), e)))
            .collect::<Result<Vec<_>>>()?;

        let mut file_refs: Vec<&mut File> = files.iter_mut().collect();
        let content = Content::from_readers(&mut file_refs)?;

        let inner = GgufDeviceMapLoaderInner {
            model: &content,
            arch: self.metadata.architecture,
        };

        inner.non_mapped_size_in_bytes("", dtype, 1, None)
    }

    /// Get model configuration for KV cache and memory planning.
    ///
    /// Returns a config implementing `ModelConfigLike` for computing:
    /// - KV cache size per sequence
    /// - Paged attention block allocation
    /// - Memory budget planning
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loader = GgufLoader::open(&["model.gguf"])?;
    /// let config = loader.model_config()?;
    /// let kv_per_token = config.num_kv_heads() * config.k_head_dim() * 2; // K + V
    /// ```
    pub fn model_config(&self) -> Result<ContentConfig> {
        let mut readers: Vec<BufReader<File>> = self
            .paths
            .iter()
            .map(|p| Ok(BufReader::new(File::open(p)?)))
            .collect::<Result<Vec<_>>>()?;

        let mut reader_refs: Vec<&mut BufReader<File>> = readers.iter_mut().collect();
        let content = Content::from_readers(&mut reader_refs)?;

        Ok(ContentConfig::from(&content))
    }
}

impl MetadataLoader for GgufLoader {
    fn architecture(&self) -> Result<String> {
        self.architecture_string()
    }

    fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.metadata.context_length.unwrap_or(4096)
    }

    fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }

    fn num_attention_heads(&self) -> usize {
        self.metadata.num_attention_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.metadata.num_kv_heads
    }

    fn tokenizer(&self) -> Result<Arc<Tokenizer>> {
        let mut readers: Vec<BufReader<File>> = self
            .paths
            .iter()
            .map(|p| Ok(BufReader::new(File::open(p)?)))
            .collect::<Result<Vec<_>>>()?;

        let mut reader_refs: Vec<&mut BufReader<File>> = readers.iter_mut().collect();
        let content = Content::from_readers(&mut reader_refs)?;

        let conversion = convert_gguf_to_hf_tokenizer(&content)?;
        Ok(Arc::new(conversion.tokenizer))
    }

    fn chat_template_string(&self) -> Result<Option<String>> {
        let mut readers: Vec<BufReader<File>> = self
            .paths
            .iter()
            .map(|p| Ok(BufReader::new(File::open(p)?)))
            .collect::<Result<Vec<_>>>()?;

        let mut reader_refs: Vec<&mut BufReader<File>> = readers.iter_mut().collect();
        let content = Content::from_readers(&mut reader_refs)?;

        get_gguf_chat_template(&content)
    }

    fn format_metadata(&self) -> &dyn LoaderMetadata {
        &self.metadata
    }
}


// ============================================================================
// Text Pipeline Loading
// ============================================================================

/// Load a text generation pipeline from GGUF files.
///
/// This is the **single dispatch point** where typed pipelines are converted to
/// trait objects. The architecture is detected from GGUF metadata, and the
/// appropriate model type is loaded with full monomorphization. The resulting
/// pipeline is then boxed for engine compatibility.
///
/// # Architecture Flow
///
/// ```text
/// GGUF file → detect arch → load model via FromGGUF → TextPipeline<M> → Box<dyn Pipeline>
///                                                           ↑
///                                               Monomorphized forward pass
/// ```
///
/// # Arguments
///
/// * `paths` - Paths to .gguf file(s). For sharded models, provide all shards.
/// * `device` - Device for model weights
/// * `attention` - Attention implementation (eager or paged)
/// * `dtype` - Data type for computations
/// * `layer_range` - Layer range for pipeline parallelism (None = all layers)
/// * `mapper` - Optional device mapper for tensor placement. If None, uses
///   `SingleDeviceMapper` to place all tensors on `device`.
///
/// # Returns
///
/// `CausalLMPipeline` - Monomorphized pipeline enum with static dispatch.
///
/// # Example
///
/// ```ignore
/// // Simple single-device loading
/// let pipeline = load_text_pipeline(
///     &["model.gguf"],
///     &Device::Cpu,
///     AttentionImplementation::Eager,
///     DType::F16,
///     None,  // all layers
///     None,  // default mapper
/// )?;
///
/// // Pipeline parallelism with layer range
/// let pipeline = load_text_pipeline(
///     &["model.gguf"],
///     &Device::Cpu,
///     AttentionImplementation::Eager,
///     DType::F16,
///     Some(0..16),  // first 16 layers only
///     None,
/// )?;
/// ```
pub fn load_text_pipeline(
    paths: &[impl AsRef<Path>],
    device: &Device,
    attention: AttentionImplementation,
    dtype: DType,
    layer_range: Option<std::ops::Range<usize>>,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
) -> Result<crate::pipeline::CausalLMPipeline> {
    use crate::pipeline::CausalLMPipeline;

    // Detect architecture from metadata
    let loader = GgufLoader::open(paths)?;
    let arch = loader.gguf_architecture();

    // Use provided mapper or default to single-device
    let mapper = mapper.unwrap_or_else(|| {
        Box::new(crate::device_map::SingleDeviceMapper::new(device.clone()))
    });

    // Dispatch based on architecture - each arm loads the model and constructs
    // a typed pipeline wrapped in the enum. All forward passes are monomorphized.
    match arch {
        GGUFArchitecture::Llama => {
            // Check for MoE: Mixtral uses Llama architecture but with experts
            if loader.metadata().expert_count > 1 {
                info!(
                    "Detected MoE model (expert_count={}), using Mixtral",
                    loader.metadata().expert_count
                );
                let pipeline = load_pipeline_for_model::<Mixtral>(&loader, device, mapper, attention, dtype, layer_range)?;
                Ok(CausalLMPipeline::Mixtral(pipeline))
            } else {
                let pipeline = load_pipeline_for_model::<LlamaModel>(&loader, device, mapper, attention, dtype, layer_range)?;
                Ok(CausalLMPipeline::Llama(pipeline))
            }
        }
        GGUFArchitecture::Mistral3 => {
            let pipeline = load_pipeline_for_model::<QMistral3>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Mistral3(pipeline))
        }
        GGUFArchitecture::Phi2 => {
            let pipeline = load_pipeline_for_model::<QPhi2>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Phi2(pipeline))
        }
        GGUFArchitecture::Phi3 => {
            let pipeline = load_pipeline_for_model::<QPhi3>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Phi3(pipeline))
        }
        GGUFArchitecture::Starcoder2 => {
            let pipeline = load_pipeline_for_model::<QStarcoder2>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Starcoder2(pipeline))
        }
        GGUFArchitecture::Qwen2 => {
            let pipeline = load_pipeline_for_model::<QQwen>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Qwen2(pipeline))
        }
        GGUFArchitecture::Qwen3 => {
            let pipeline = load_pipeline_for_model::<QQwen3>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Qwen3(pipeline))
        }
        GGUFArchitecture::Qwen3MoE => {
            let pipeline = load_pipeline_for_model::<QQwen3MoE>(&loader, device, mapper, attention, dtype, layer_range)?;
            Ok(CausalLMPipeline::Qwen3MoE(pipeline))
        }
        // Unsupported architectures
        arch => bail!(
            "Unsupported GGUF architecture `{arch}` for text pipeline. \
             Supported: Llama, Mistral3, Phi2, Phi3, Starcoder2, Qwen2, Qwen3, Qwen3MoE"
        ),
    }
}

/// Load a model and construct a TextPipeline for a specific model type.
///
/// This is a helper function that does the actual work:
/// 1. Opens GGUF content
/// 2. Loads model via FromGGUF trait
/// 3. Extracts tokenizer and chat template
/// 4. Constructs TextPipeline<M>
fn load_pipeline_for_model<M: LanguageModel + FromGGUF + Send + Sync + 'static>(
    loader: &GgufLoader,
    device: &Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    attention: AttentionImplementation,
    dtype: DType,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<crate::pipeline::TextPipeline<M>> {
    use crate::pipeline::chat_template::{ChatTemplate, ChatTemplateValue};
    use crate::pipeline::{GeneralMetadata, ModelKind, Modalities, SupportedModality};
    use either::Either;

    // Open files and parse content
    let mut readers: Vec<BufReader<File>> = loader
        .paths
        .iter()
        .map(|p| Ok(BufReader::new(File::open(p)?)))
        .collect::<Result<Vec<_>>>()?;

    let mut reader_refs: Vec<&mut BufReader<File>> = readers.iter_mut().collect();
    let content = Content::from_readers(&mut reader_refs)?;

    // Load model via FromGGUF trait
    let model = M::from_gguf(content, device, mapper, attention, dtype, layer_range)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    // Get tokenizer
    let tokenizer = loader.tokenizer()?;

    // Get chat template
    let template_string = loader.chat_template_string()?;
    let mut chat_template = ChatTemplate::default();
    chat_template.chat_template = template_string.map(|s| ChatTemplateValue(Either::Left(s)));

    // Build model ID from path
    let model_id = loader
        .paths
        .first()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Build metadata
    let metadata = Arc::new(GeneralMetadata {
        max_seq_len: model.max_seq_len(),
        llg_factory: None,
        no_kv_cache: false,
        no_prefix_cache: false,
        num_hidden_layers: model.num_layers(),
        eos_tok: vec![],
        kind: ModelKind::GgufQuantized {
            quant: crate::pipeline::QuantizationKind::Gguf,
        },
        is_xlora: false,
        activation_dtype: dtype,
        sliding_window: None,
        cache_config: None,
        cache_engine: None,
        model_metadata: None,
        modalities: Modalities {
            input: vec![SupportedModality::Text],
            output: vec![SupportedModality::Text],
        },
    });

    // Create a mapper for the pipeline (the real mapper was consumed by model loading)
    let pipeline_mapper: Box<dyn DeviceMapper + Send + Sync> =
        Box::new(crate::device_map::SingleDeviceMapper::new(device.clone()));

    Ok(crate::pipeline::TextPipeline::new(
        model,
        tokenizer,
        Arc::new(chat_template),
        model_id,
        pipeline_mapper,
        metadata,
    ))
}

// =============================================================================
// CausalLMLoaderBuilder - Builder pattern for loading causal LM pipelines
// =============================================================================

/// Source specification for model loading.
#[derive(Clone)]
enum ModelSource {
    /// Local filesystem paths to GGUF file(s)
    LocalGguf(Vec<PathBuf>),
    /// GGUF files from HuggingFace Hub
    HuggingFaceGguf {
        repo_id: String,
        filenames: Vec<String>,
        revision: Option<String>,
    },
    /// Safetensors model from HuggingFace Hub (config.json + *.safetensors)
    HuggingFaceSafetensors {
        repo_id: String,
        revision: Option<String>,
    },
}

/// Model metadata extracted from safetensors config.json for device mapping and paged attention.
///
/// This struct uses serde renames to deserialize from the standard HF config format,
/// which allows us to extract the metadata before knowing the specific model type.
/// The metadata is used for:
/// - Computing layer sizes for automatic device mapping
/// - Configuring paged attention cache dimensions
#[derive(serde::Deserialize)]
struct SafetensorsModelMetadata {
    // Core dimensions
    hidden_size: usize,
    intermediate_size: usize,
    #[serde(rename = "num_hidden_layers")]
    num_layers: usize,
    vocab_size: usize,

    // Attention config
    #[serde(rename = "num_attention_heads")]
    num_attn_heads: usize,
    #[serde(rename = "num_key_value_heads")]
    num_kv_heads: usize,
    // Note: head_dim is computed, not deserialized
    #[serde(skip)]
    head_dim: usize,
    sliding_window: Option<usize>,

    // Whether output weights are tied to embeddings
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
}

fn default_tie_word_embeddings() -> bool {
    false
}

impl SafetensorsModelMetadata {
    /// Parse from config.json string, computing derived fields.
    fn from_config_json(config_str: &str) -> Result<Self> {
        let mut meta: Self = serde_json::from_str(config_str)?;
        // Compute head_dim (this is hidden_size / num_attention_heads in all Llama-family models)
        meta.head_dim = meta.hidden_size / meta.num_attn_heads;
        Ok(meta)
    }
}

impl SafetensorsModelMetadata {
    /// Compute per-layer weight size in elements (before dtype multiplication).
    fn layer_size_elems(&self, weight_pack_factor: usize) -> usize {
        // RMS norms (2 per layer)
        let input_layernorm = self.hidden_size;
        let post_attention_layernorm = self.hidden_size;

        // Attention projections
        let size_q = self.head_dim * self.num_attn_heads;
        let size_kv = self.head_dim * self.num_kv_heads;
        let q_proj = self.hidden_size * size_q / weight_pack_factor;
        let k_proj = self.hidden_size * size_kv / weight_pack_factor;
        let v_proj = self.hidden_size * size_kv / weight_pack_factor;
        let o_proj = size_q * self.hidden_size / weight_pack_factor;

        // MLP projections (gated)
        let gate_proj = self.hidden_size * self.intermediate_size / weight_pack_factor;
        let up_proj = self.hidden_size * self.intermediate_size / weight_pack_factor;
        let down_proj = self.intermediate_size * self.hidden_size / weight_pack_factor;

        input_layernorm + post_attention_layernorm
            + q_proj + k_proj + v_proj + o_proj
            + gate_proj + up_proj + down_proj
    }

    /// Compute non-mapped (embedding + output) size in elements.
    fn non_mapped_size_elems(&self) -> usize {
        let token_embd = self.vocab_size * self.hidden_size;
        let output_norm = self.hidden_size;
        let output = if self.tie_word_embeddings {
            0 // Tied to token_embd
        } else {
            self.vocab_size * self.hidden_size
        };
        token_embd + output_norm + output
    }
}

impl crate::DeviceMappedModelLoader for SafetensorsModelMetadata {
    fn mapped_max_act_size_elems(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> anyhow::Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        Ok(max_batch_size * self.num_attn_heads * max_seq_len.min(&crate::attention::ATTENTION_CHUNK_SIZE))
    }

    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        _config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&crate::matformer::MatformerSliceConfig>,
    ) -> anyhow::Result<usize> {
        Ok(self.non_mapped_size_elems() * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        _config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&crate::matformer::MatformerSliceConfig>,
    ) -> anyhow::Result<Vec<usize>> {
        let per_layer = self.layer_size_elems(weight_pack_factor) * dtype.size_in_bytes();
        Ok(vec![per_layer; self.num_layers])
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        None
    }

    fn num_layers(&self, _config: &str) -> anyhow::Result<usize> {
        Ok(self.num_layers)
    }

    fn model_config(&self, _config: &str) -> anyhow::Result<Box<dyn ModelConfigLike>> {
        Ok(Box::new(ModelConfigMetadata {
            max_seq_len: 0, // Will be filled from model
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_kv_heads: self.num_kv_heads,
            num_attn_heads: self.num_attn_heads,
            sliding_window: self.sliding_window,
            k_head_dim: self.head_dim,
            v_head_dim: self.head_dim,
        }))
    }
}

/// Builder for loading causal language model pipelines.
///
/// Provides a fluent API for configuring and loading GGUF models with support for:
/// - Local files or HuggingFace Hub downloads
/// - Paged attention for memory efficiency
/// - Multi-GPU device mapping
/// - Pipeline parallelism via layer ranges
/// - Custom chat templates
///
/// # Example
///
/// ```ignore
/// use mistralrs_core::{CausalLMLoaderBuilder, Device, DType};
///
/// // Load GGUF from local file
/// let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&["model.gguf"])
///     .with_device(Device::Cpu)
///     .with_dtype(DType::F32)
///     .build()?;
///
/// // Load GGUF from HuggingFace Hub
/// let pipeline = CausalLMLoaderBuilder::from_hf_gguf("unsloth/Qwen3-0.6B-GGUF", &["Qwen3-0.6B-Q4_K_M.gguf"])
///     .with_device(Device::new_cuda(0)?)
///     .with_dtype(DType::F16)
///     .build()?;
///
/// // Load safetensors from HuggingFace Hub
/// let pipeline = CausalLMLoaderBuilder::from_hf_safetensors("Qwen/Qwen2-0.5B")
///     .with_device(Device::new_cuda(0)?)
///     .with_dtype(DType::BF16)
///     .build()?;
///
/// // With paged attention
/// let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&["model.gguf"])
///     .with_device(Device::new_cuda(0)?)
///     .with_paged_attention(PagedAttentionConfig::default())
///     .build()?;
/// ```
pub struct CausalLMLoaderBuilder {
    /// Model source (local paths or HF Hub)
    source: ModelSource,
    /// Target device for model weights
    device: Device,
    /// Data type for computations
    dtype: DType,
    /// Attention implementation (eager or paged)
    attention: AttentionImplementation,
    /// Layer range for pipeline parallelism (None = all layers)
    layer_range: Option<Range<usize>>,
    /// Paged attention configuration
    paged_attn_config: Option<PagedAttentionConfig>,
    /// Device mapping configuration
    device_map_setting: DeviceMapSetting,
    /// Explicit chat template string
    chat_template: Option<String>,
    /// HuggingFace token source for authenticated downloads
    token_source: TokenSource,
    /// Whether to disable KV cache
    no_kv_cache: bool,
    /// Whether to suppress progress output
    silent: bool,
    /// Optional tokenizer model ID (for sourcing tokenizer from different HF repo)
    tok_model_id: Option<String>,
    /// Optional topology for device mapping
    topology: Option<crate::Topology>,
    /// Optional explicit Jinja template file path
    jinja_explicit: Option<String>,
    /// In-situ quantization type (for safetensors loading only)
    isq: Option<IsqType>,
    /// LoRA adapter repo IDs (from HuggingFace Hub)
    lora_adapters: Vec<String>,
}

impl CausalLMLoaderBuilder {
    /// Create a builder for loading from local GGUF files.
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to .gguf file(s). For sharded models, provide all shards.
    pub fn from_gguf_paths(paths: &[impl AsRef<Path>]) -> Self {
        let paths: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
        Self {
            source: ModelSource::LocalGguf(paths),
            device: Device::Cpu,
            dtype: DType::F32,
            attention: AttentionImplementation::Eager,
            layer_range: None,
            paged_attn_config: None,
            device_map_setting: DeviceMapSetting::dummy(),
            chat_template: None,
            token_source: TokenSource::CacheToken,
            no_kv_cache: false,
            silent: false,
            tok_model_id: None,
            topology: None,
            jinja_explicit: None,
            isq: None,
            lora_adapters: Vec::new(),
        }
    }

    /// Create a builder for loading GGUF from HuggingFace Hub.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "unsloth/Qwen3-0.6B-GGUF")
    /// * `filenames` - GGUF filename(s) within the repository
    pub fn from_hf_gguf(repo_id: &str, filenames: &[&str]) -> Self {
        Self {
            source: ModelSource::HuggingFaceGguf {
                repo_id: repo_id.to_string(),
                filenames: filenames.iter().map(|s| s.to_string()).collect(),
                revision: None,
            },
            device: Device::Cpu,
            dtype: DType::F32,
            attention: AttentionImplementation::Eager,
            layer_range: None,
            paged_attn_config: None,
            device_map_setting: DeviceMapSetting::dummy(),
            chat_template: None,
            token_source: TokenSource::CacheToken,
            no_kv_cache: false,
            silent: false,
            tok_model_id: None,
            topology: None,
            jinja_explicit: None,
            isq: None,
            lora_adapters: Vec::new(),
        }
    }

    /// Create a builder for loading safetensors model from HuggingFace Hub.
    ///
    /// This automatically downloads config.json, tokenizer files, and safetensors
    /// weights from the repository.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID (e.g., "Qwen/Qwen2-0.5B")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = CausalLMLoaderBuilder::from_hf_safetensors("Qwen/Qwen2-0.5B")
    ///     .with_device(Device::new_cuda(0)?)
    ///     .with_dtype(DType::BF16)
    ///     .build()?;
    /// ```
    pub fn from_hf_safetensors(repo_id: &str) -> Self {
        Self {
            source: ModelSource::HuggingFaceSafetensors {
                repo_id: repo_id.to_string(),
                revision: None,
            },
            device: Device::Cpu,
            dtype: DType::F32,
            attention: AttentionImplementation::Eager,
            layer_range: None,
            paged_attn_config: None,
            device_map_setting: DeviceMapSetting::dummy(),
            chat_template: None,
            token_source: TokenSource::CacheToken,
            no_kv_cache: false,
            silent: false,
            tok_model_id: None,
            topology: None,
            jinja_explicit: None,
            isq: None,
            lora_adapters: Vec::new(),
        }
    }

    /// Deprecated: Use `from_gguf_paths` instead.
    #[deprecated(since = "0.8.0", note = "Renamed to from_gguf_paths for clarity")]
    pub fn from_paths(paths: &[impl AsRef<Path>]) -> Self {
        Self::from_gguf_paths(paths)
    }

    /// Deprecated: Use `from_hf_gguf` instead.
    #[deprecated(since = "0.8.0", note = "Renamed to from_hf_gguf for clarity")]
    pub fn from_hf(repo_id: &str, filenames: &[&str]) -> Self {
        Self::from_hf_gguf(repo_id, filenames)
    }

    /// Set the target device for model weights.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the data type for computations.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the attention implementation.
    pub fn with_attention(mut self, attention: AttentionImplementation) -> Self {
        self.attention = attention;
        self
    }

    /// Set the layer range for pipeline parallelism.
    ///
    /// When set, only the specified layers are loaded. This enables
    /// distributing a model across multiple nodes.
    pub fn with_layer_range(mut self, range: Range<usize>) -> Self {
        self.layer_range = Some(range);
        self
    }

    /// Enable paged attention for memory-efficient KV cache.
    pub fn with_paged_attention(mut self, config: PagedAttentionConfig) -> Self {
        self.paged_attn_config = Some(config);
        self.attention = AttentionImplementation::PagedAttention;
        self
    }

    /// Set the device mapping configuration.
    ///
    /// Use `DeviceMapSetting::Auto` for automatic multi-GPU distribution.
    pub fn with_device_map(mut self, setting: DeviceMapSetting) -> Self {
        self.device_map_setting = setting;
        self
    }

    /// Set an explicit chat template string.
    pub fn with_chat_template(mut self, template: String) -> Self {
        self.chat_template = Some(template);
        self
    }

    /// Set the HuggingFace token source for authenticated downloads.
    pub fn with_token_source(mut self, source: TokenSource) -> Self {
        self.token_source = source;
        self
    }

    /// Set the HuggingFace revision (branch/tag/commit).
    pub fn with_revision(mut self, revision: String) -> Self {
        match &mut self.source {
            ModelSource::HuggingFaceGguf { revision: ref mut rev, .. } => {
                *rev = Some(revision);
            }
            ModelSource::HuggingFaceSafetensors { revision: ref mut rev, .. } => {
                *rev = Some(revision);
            }
            ModelSource::LocalGguf(_) => {}
        }
        self
    }

    /// Disable KV cache (for debugging).
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }

    /// Suppress progress output during loading.
    pub fn silent(mut self) -> Self {
        self.silent = true;
        self
    }

    /// Set an optional tokenizer model ID.
    ///
    /// When set, the tokenizer will be loaded from this HuggingFace repo
    /// instead of being extracted from the GGUF file. This is useful for
    /// GGUF files that don't have an embedded tokenizer.
    pub fn with_tok_model_id(mut self, tok_model_id: String) -> Self {
        self.tok_model_id = Some(tok_model_id);
        self
    }

    /// Set the device mapping topology.
    ///
    /// This provides additional control over how model layers are mapped
    /// to devices. Used in conjunction with device mapping settings.
    pub fn with_topology(mut self, topology: crate::Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Set an explicit Jinja template file path.
    ///
    /// When set, the chat template will be loaded from this file
    /// instead of using the embedded template in the GGUF file.
    pub fn with_jinja_explicit(mut self, jinja_path: String) -> Self {
        self.jinja_explicit = Some(jinja_path);
        self
    }

    /// Enable in-situ quantization (ISQ) for safetensors loading.
    ///
    /// When set, model weights will be quantized during loading to the specified
    /// quantization type. This reduces memory usage but only applies to safetensors
    /// models (GGUF models are already quantized).
    ///
    /// # Arguments
    ///
    /// * `isq_type` - The quantization type (e.g., Q4K, Q8_0, HQQ4, AFQ4)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use mistralrs_quant::IsqType;
    ///
    /// let pipeline = CausalLMLoaderBuilder::from_hf_safetensors("Qwen/Qwen2-0.5B")
    ///     .with_device(Device::new_cuda(0)?)
    ///     .with_isq(IsqType::Q4K)
    ///     .build()?;
    /// ```
    pub fn with_isq(mut self, isq_type: IsqType) -> Self {
        self.isq = Some(isq_type);
        self
    }

    /// Add a LoRA adapter to be merged during model loading.
    ///
    /// LoRA (Low-Rank Adaptation) adapters allow fine-tuned behavior without
    /// full model fine-tuning. Adapters are merged into the base model weights
    /// at load time for zero inference overhead.
    ///
    /// This method can be called multiple times to merge multiple adapters.
    /// Adapters are always loaded from HuggingFace Hub in safetensors format,
    /// regardless of whether the base model is GGUF or safetensors.
    ///
    /// # Arguments
    ///
    /// * `adapter_repo_id` - HuggingFace repository ID for the adapter
    ///   (e.g., "username/my-lora-adapter")
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Single adapter
    /// let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&["model.gguf"])
    ///     .with_lora_adapter("username/my-lora-adapter")
    ///     .build()?;
    ///
    /// // Multiple adapters (merged in order)
    /// let pipeline = CausalLMLoaderBuilder::from_hf_safetensors("Qwen/Qwen2-0.5B")
    ///     .with_lora_adapter("adapter1/repo")
    ///     .with_lora_adapter("adapter2/repo")
    ///     .build()?;
    /// ```
    ///
    /// # Note
    ///
    /// Adapters must be compatible with the base model architecture.
    /// The adapter's `target_modules` in `adapter_config.json` determines
    /// which layers receive the LoRA modifications.
    pub fn with_lora_adapter(mut self, adapter_repo_id: impl Into<String>) -> Self {
        self.lora_adapters.push(adapter_repo_id.into());
        self
    }

    /// Build the pipeline, loading the model.
    ///
    /// This downloads the model (if from HF Hub), detects the architecture,
    /// and constructs the appropriate monomorphized pipeline.
    ///
    /// Returns `CausalLMPipeline` - a monomorphized enum with static dispatch.
    /// For engine integration, use `build_async()` which returns `Arc<Mutex<dyn Pipeline>>`.
    pub fn build(self) -> Result<crate::pipeline::CausalLMPipeline> {
        match &self.source {
            ModelSource::LocalGguf(_) | ModelSource::HuggingFaceGguf { .. } => {
                self.build_from_gguf()
            }
            ModelSource::HuggingFaceSafetensors { .. } => {
                self.build_from_safetensors()
            }
        }
    }

    /// Build pipeline from GGUF source.
    fn build_from_gguf(self) -> Result<crate::pipeline::CausalLMPipeline> {
        // Resolve paths (download from HF if needed)
        let paths = self.resolve_gguf_paths()?;

        // Detect architecture and load
        let loader = GgufLoader::open(&paths)?;
        let arch = loader.gguf_architecture();

        info!("Loading {} model from {:?}", arch, paths);

        // Load LoRA adapters before model construction
        self.load_lora_adapters()?;

        // Dispatch based on architecture (device mapping computed inside with full Content)
        let result = self.build_for_architecture(arch, &loader);

        // Clear LoRA adapters after loading
        self.clear_lora_adapters();

        result
    }

    /// Build pipeline from safetensors source.
    fn build_from_safetensors(self) -> Result<crate::pipeline::CausalLMPipeline> {
        use crate::pipeline::loaders::NormalLoaderType;

        let (repo_id, revision) = match &self.source {
            ModelSource::HuggingFaceSafetensors { repo_id, revision } => {
                (repo_id.clone(), revision.clone())
            }
            _ => bail!("build_from_safetensors called with non-safetensors source"),
        };

        if !self.silent {
            info!("Downloading safetensors model from HuggingFace: {}", repo_id);
        }

        // Download files from HuggingFace
        let api = Api::new().map_err(|e| anyhow!("Failed to create HF API: {}", e))?;
        let repo = match &revision {
            Some(rev) => api.repo(Repo::with_revision(
                repo_id.clone(),
                RepoType::Model,
                rev.clone(),
            )),
            None => api.repo(Repo::new(repo_id.clone(), RepoType::Model)),
        };

        // Download config.json
        let config_path = repo
            .get("config.json")
            .map_err(|e| anyhow!("Failed to download config.json: {}", e))?;
        let config_str = std::fs::read_to_string(&config_path)?;

        // Detect architecture from config.json
        #[derive(serde::Deserialize)]
        struct ArchConfig {
            architectures: Vec<String>,
        }
        let arch_config: ArchConfig = serde_json::from_str(&config_str)?;
        if arch_config.architectures.len() != 1 {
            bail!("Expected exactly one architecture in config.json");
        }
        let arch_name = &arch_config.architectures[0];
        let loader_type = NormalLoaderType::from_causal_lm_name(arch_name)?;

        info!("Detected architecture: {} ({:?})", arch_name, loader_type);

        // Download safetensors files
        let safetensors_paths = self.download_safetensors_files(&repo, &repo_id)?;

        // Download tokenizer files
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow!("Failed to download tokenizer.json: {}", e))?;

        // Load LoRA adapters before model construction
        self.load_lora_adapters()?;

        // Dispatch based on architecture
        let result = self.build_for_safetensors_architecture(
            loader_type,
            &config_str,
            &safetensors_paths,
            &tokenizer_path,
            &repo,
            &repo_id,
        );

        // Clear LoRA adapters after loading
        self.clear_lora_adapters();

        result
    }

    /// Download safetensors files from a HuggingFace repo.
    fn download_safetensors_files(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        repo_id: &str,
    ) -> Result<Vec<PathBuf>> {
        // List files in repo to find safetensors
        let files = repo.info().map_err(|e| anyhow!("Failed to get repo info: {}", e))?;

        let safetensor_files: Vec<_> = files
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".safetensors"))
            .map(|s| s.rfilename.clone())
            .collect();

        if safetensor_files.is_empty() {
            bail!("No .safetensors files found in {}", repo_id);
        }

        if !self.silent {
            info!("Found {} safetensors files", safetensor_files.len());
        }

        let mut paths = Vec::new();
        for filename in &safetensor_files {
            let path = repo.get(filename).map_err(|e| {
                anyhow!("Failed to download {}/{}: {}", repo_id, filename, e)
            })?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Load and register LoRA adapters for merging during model loading.
    ///
    /// Downloads adapter weights from HuggingFace Hub and registers them
    /// via `push_applied_lora()`. The adapters will be merged into base
    /// model weights during VarBuilder tensor loading.
    fn load_lora_adapters(&self) -> Result<()> {
        use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

        if self.lora_adapters.is_empty() {
            return Ok(());
        }

        info!("Loading {} LoRA adapter(s)", self.lora_adapters.len());

        let api = Api::new().map_err(|e| anyhow!("Failed to create HF API: {}", e))?;

        for adapter_repo_id in &self.lora_adapters {
            if !self.silent {
                info!("Downloading LoRA adapter: {}", adapter_repo_id);
            }

            let repo = api.repo(Repo::new(adapter_repo_id.clone(), RepoType::Model));

            // Download adapter config
            let config_path = repo
                .get("adapter_config.json")
                .map_err(|e| anyhow!("Failed to download adapter_config.json from {}: {}", adapter_repo_id, e))?;
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: mistralrs_quant::LoraConfig = serde_json::from_str(&config_str)
                .map_err(|e| anyhow!("Failed to parse adapter_config.json from {}: {}", adapter_repo_id, e))?;

            // Download adapter weights
            let weights_path = repo
                .get("adapter_model.safetensors")
                .map_err(|e| anyhow!("Failed to download adapter_model.safetensors from {}: {}", adapter_repo_id, e))?;

            // Create VarBuilder for adapter weights
            let weights = from_mmaped_safetensors(
                vec![weights_path],
                Vec::new(), // xlora_paths
                Some(self.dtype),
                &self.device,
                Vec::new(), // layer_devices
                true, // silent for adapter loading
                None, // make_dummy_regexes
                |_| true, // predicate
                Arc::new(move |_| DeviceForLoadTensor::Base),
            )?;

            info!(
                "Loaded LoRA adapter '{}' (rank={}, alpha={}, targets={:?})",
                adapter_repo_id, config.rank, config.alpha, config.target_modules
            );

            // Register adapter for merging during model loading
            mistralrs_quant::push_applied_lora(mistralrs_quant::LoraAdapter { config, weights });
        }

        Ok(())
    }

    /// Clear registered LoRA adapters after model loading.
    fn clear_lora_adapters(&self) {
        if !self.lora_adapters.is_empty() {
            mistralrs_quant::clear_applied_loras();
        }
    }

    /// Build pipeline for detected safetensors architecture.
    fn build_for_safetensors_architecture(
        &self,
        loader_type: crate::pipeline::loaders::NormalLoaderType,
        config_str: &str,
        safetensors_paths: &[PathBuf],
        tokenizer_path: &Path,
        repo: &hf_hub::api::sync::ApiRepo,
        repo_id: &str,
    ) -> Result<crate::pipeline::CausalLMPipeline> {
        use crate::pipeline::CausalLMPipeline;
        use crate::pipeline::loaders::NormalLoaderType;
        use crate::utils::model_config::FromSafetensors;
        use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

        // Extract model metadata early for device mapping decisions
        let model_meta = SafetensorsModelMetadata::from_config_json(config_str)?;

        // Create device mapper using model metadata
        let mapper = self.create_device_mapper_for_safetensors(&model_meta)?;

        // Set up immediate ISQ if configured
        if let Some(isq_type) = self.isq {
            let predicates = self.get_isq_predicates_for_loader_type(&loader_type)?;
            if predicates.is_empty() {
                warn!("No ISQ predicates for {:?}, ISQ will not be applied", loader_type);
            } else {
                info!("Setting up immediate ISQ ({:?}) with {} predicates", isq_type, predicates.len());
                mistralrs_quant::set_immediate_isq(Some(isq_type), predicates);
            }
        }

        // Create VarBuilder from safetensors files
        // Arguments: paths, xlora_paths, dtype, base_device, layer_devices, silent,
        //            make_dummy_regexes, predicate, get_device_for_tensor
        let vb = from_mmaped_safetensors(
            safetensors_paths.to_vec(),
            Vec::new(), // xlora_paths
            Some(self.dtype),
            &self.device,
            Vec::new(), // layer_devices (all on base device)
            self.silent,
            None, // make_dummy_regexes
            |_| true, // predicate - include all tensors
            Arc::new(move |_| DeviceForLoadTensor::Base), // get_device_for_tensor
        )?;

        let result = match loader_type {
            NormalLoaderType::Qwen2 => {
                let config: crate::models::quantized_qwen::Config =
                    serde_json::from_str(config_str)?;
                let model = QQwen::from_safetensors(
                    &config,
                    vb,
                    &self.device,
                    mapper,
                    self.attention,
                    self.dtype,
                    self.layer_range.clone(),
                )?;
                let pipeline = self.build_text_pipeline_safetensors(
                    model,
                    &model_meta,
                    tokenizer_path,
                    repo,
                    repo_id,
                )?;
                Ok(CausalLMPipeline::Qwen2(pipeline))
            }
            NormalLoaderType::Phi3 => {
                let config: crate::models::quantized_phi3::Config =
                    serde_json::from_str(config_str)?;
                let model = QPhi3::from_safetensors(
                    &config,
                    vb,
                    &self.device,
                    mapper,
                    self.attention,
                    self.dtype,
                    self.layer_range.clone(),
                )?;
                let pipeline = self.build_text_pipeline_safetensors(
                    model,
                    &model_meta,
                    tokenizer_path,
                    repo,
                    repo_id,
                )?;
                Ok(CausalLMPipeline::Phi3(pipeline))
            }
            NormalLoaderType::Qwen3 => {
                let config: crate::models::quantized_qwen3::Config =
                    serde_json::from_str(config_str)?;
                let model = QQwen3::from_safetensors(
                    &config,
                    vb,
                    &self.device,
                    mapper,
                    self.attention,
                    self.dtype,
                    self.layer_range.clone(),
                )?;
                let pipeline = self.build_text_pipeline_safetensors(
                    model,
                    &model_meta,
                    tokenizer_path,
                    repo,
                    repo_id,
                )?;
                Ok(CausalLMPipeline::Qwen3(pipeline))
            }
            NormalLoaderType::Mistral => {
                // Use QMistral3 which supports both regular Mistral and YaRN-scaled Mistral3
                let config: crate::models::quantized_mistral3::Config =
                    serde_json::from_str(config_str)?;
                let model = QMistral3::from_safetensors(
                    &config,
                    vb,
                    &self.device,
                    mapper,
                    self.attention,
                    self.dtype,
                    self.layer_range.clone(),
                )?;
                let pipeline = self.build_text_pipeline_safetensors(
                    model,
                    &model_meta,
                    tokenizer_path,
                    repo,
                    repo_id,
                )?;
                Ok(CausalLMPipeline::Mistral3(pipeline))
            }
            _ => bail!(
                "Safetensors loading not yet supported for {:?}. \
                 Currently supported: Mistral, Qwen2, Qwen3, Phi3. \
                 Use GGUF format or contribute a FromSafetensors implementation.",
                loader_type
            ),
        };

        // Clear immediate ISQ after loading
        if self.isq.is_some() {
            mistralrs_quant::clear_immediate_isq();
        }

        result
    }

    /// Get ISQ predicates for a specific architecture.
    ///
    /// Returns regex patterns that match the weight tensor names to be quantized.
    fn get_isq_predicates_for_loader_type(
        &self,
        loader_type: &crate::pipeline::loaders::NormalLoaderType,
    ) -> Result<Vec<Regex>> {
        use crate::pipeline::loaders::NormalLoaderType;

        match loader_type {
            NormalLoaderType::Qwen2 | NormalLoaderType::Qwen3 | NormalLoaderType::Mistral => Ok(vec![
                Regex::new(r"lm_head\.(weight|bias)$")?,
                // Attention
                Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
                // MLP
                Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            ]),
            NormalLoaderType::Phi3 => Ok(vec![
                Regex::new(r"lm_head\.(weight|bias)$")?,
                // Attention (fused qkv)
                Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
                // MLP (fused gate_up)
                Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            ]),
            _ => Ok(Vec::new()),
        }
    }

    /// Create device mapper for safetensors model.
    fn create_device_mapper_for_safetensors(
        &self,
        model_meta: &SafetensorsModelMetadata,
    ) -> Result<Box<dyn crate::device_map::DeviceMapper + Send + Sync>> {
        use crate::device_map::SingleDeviceMapper;
        use crate::pipeline::loaders::auto_device_map;

        let mut mapper = self.device_map_setting.clone();

        if let DeviceMapSetting::Auto(params) = &self.device_map_setting {
            // Get all available devices
            let devices = device_map::get_all_similar_devices(&self.device)?;
            if devices.is_empty() {
                bail!("No devices available for auto device mapping");
            }

            // Compute layer sizes and total model size
            let weight_pack_factor = 1; // safetensors are unpacked
            let layer_sizes = model_meta.layer_sizes_in_bytes(
                "", // config_str not needed for SafetensorsModelMetadata
                self.dtype,
                weight_pack_factor,
                None, // no matformer
            )?;
            let non_mapped_size = model_meta.non_mapped_size_in_bytes(
                "",
                self.dtype,
                weight_pack_factor,
                None,
            )?;
            let total_size: usize = layer_sizes.iter().sum::<usize>() + non_mapped_size;

            // Compute optimal device distribution
            let device_map_meta = auto_device_map::get_device_layers(
                model_meta,
                "", // config_str not needed
                model_meta.num_layers,
                layer_sizes,
                non_mapped_size,
                total_size,
                &devices,
                self.dtype,
                params,
                self.paged_attn_config.as_ref(),
            )?;

            info!("Auto device mapping computed for safetensors");

            // Convert to Map setting for into_mapper()
            mapper = DeviceMapSetting::Map(device_map_meta);
        }

        match &mapper {
            DeviceMapSetting::Map(_) => {
                info!("Using device mapping for safetensors");
                Ok(mapper.into_mapper(model_meta.num_layers, &self.device, None)?)
            }
            _ => Ok(Box::new(SingleDeviceMapper::new(self.device.clone()))),
        }
    }

    /// Build TextPipeline for safetensors model.
    fn build_text_pipeline_safetensors<M: LanguageModel + Send + Sync + 'static>(
        &self,
        model: M,
        model_meta: &SafetensorsModelMetadata,
        tokenizer_path: &Path,
        repo: &hf_hub::api::sync::ApiRepo,
        repo_id: &str,
    ) -> Result<crate::pipeline::TextPipeline<M>> {
        use crate::device_map::SingleDeviceMapper;
        use crate::pipeline::chat_template::{ChatTemplate, ChatTemplateValue};
        use crate::pipeline::{GeneralMetadata, ModelKind, Modalities, SupportedModality};
        use either::Either;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Build llg factory for grammar constraints (needs owned Tokenizer)
        let llg_factory = build_llg_factory(tokenizer.clone())?;

        let tokenizer = Arc::new(tokenizer);

        // Load chat template from tokenizer_config.json
        let mut chat_template = ChatTemplate::default();
        if let Some(ref explicit_template) = self.chat_template {
            chat_template.chat_template = Some(ChatTemplateValue(Either::Left(explicit_template.clone())));
        } else if let Ok(tokenizer_config_path) = repo.get("tokenizer_config.json") {
            if let Ok(tokenizer_config_str) = std::fs::read_to_string(&tokenizer_config_path) {
                #[derive(serde::Deserialize)]
                struct TokenizerConfig {
                    chat_template: Option<String>,
                }
                if let Ok(config) = serde_json::from_str::<TokenizerConfig>(&tokenizer_config_str) {
                    if let Some(template) = config.chat_template {
                        chat_template.chat_template = Some(ChatTemplateValue(Either::Left(template)));
                    }
                }
            }
        }

        // Calculate EOS tokens
        let eos_tokens = calculate_eos_tokens(&chat_template, None, &tokenizer);

        // Build model ID
        let model_id = self.tok_model_id.clone().unwrap_or_else(|| repo_id.to_string());

        // Build metadata
        let num_hidden_layers = model.num_layers();
        let max_seq_len = model.max_seq_len();

        // Setup paged attention if configured
        let (cache_config, cache_engine) = if let Some(ref paged_config) = self.paged_attn_config {
            // Adjust num_layers for pipeline parallelism
            let effective_num_layers = self.layer_range.as_ref()
                .map(|r| r.len())
                .unwrap_or(num_hidden_layers);

            let model_config = ModelConfigMetadata {
                max_seq_len,
                num_layers: effective_num_layers,
                hidden_size: model_meta.hidden_size,
                num_kv_heads: model_meta.num_kv_heads,
                num_attn_heads: model_meta.num_attn_heads,
                sliding_window: model_meta.sliding_window,
                k_head_dim: model_meta.head_dim,
                v_head_dim: model_meta.head_dim,
            };

            info!(
                "Setting up PagedAttention for safetensors: {} layers, {} KV heads, {} head dim",
                effective_num_layers,
                model_config.num_kv_heads,
                model_config.k_head_dim
            );

            // For now, use single device for layer mapping
            let layer_devices: Vec<Option<Device>> = vec![Some(self.device.clone()); effective_num_layers];

            let config = calculate_cache_config(
                paged_config.mem_gpu,
                paged_config.block_size,
                self.dtype,
                paged_config.cache_type,
                &model_config,
                &self.device,
                &layer_devices,
                self.silent,
            )?;

            let engine = CacheEngine::new(
                &model_config,
                &config,
                self.dtype,
                &self.device,
                layer_devices,
            )?;

            (Some(config), Some(engine))
        } else {
            (None, None)
        };

        // Build model config metadata for GeneralMetadata
        let model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>> = Some(Arc::new(ModelConfigMetadata {
            max_seq_len,
            num_layers: num_hidden_layers,
            hidden_size: model_meta.hidden_size,
            num_kv_heads: model_meta.num_kv_heads,
            num_attn_heads: model_meta.num_attn_heads,
            sliding_window: model_meta.sliding_window,
            k_head_dim: model_meta.head_dim,
            v_head_dim: model_meta.head_dim,
        }));

        let metadata = Arc::new(GeneralMetadata {
            max_seq_len,
            llg_factory: Some(llg_factory),
            no_kv_cache: self.no_kv_cache,
            no_prefix_cache: false,
            num_hidden_layers,
            eos_tok: eos_tokens,
            kind: ModelKind::Normal,
            is_xlora: false,
            activation_dtype: self.dtype,
            sliding_window: model_meta.sliding_window,
            cache_config,
            cache_engine,
            model_metadata,
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
        });

        // Create pipeline mapper (simplified for now)
        let pipeline_mapper: Box<dyn crate::device_map::DeviceMapper + Send + Sync> =
            Box::new(SingleDeviceMapper::new(self.device.clone()));

        Ok(crate::pipeline::TextPipeline::new(
            model,
            tokenizer,
            Arc::new(chat_template),
            model_id,
            pipeline_mapper,
            metadata,
        ))
    }

    /// Build the pipeline wrapped for engine integration.
    ///
    /// Returns `Arc<Mutex<dyn Pipeline>>` suitable for use with `MistralRsBuilder`
    /// and the engine infrastructure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&["model.gguf"])
    ///     .with_device(Device::Cpu)
    ///     .build_async()?;
    ///
    /// // Use with engine
    /// let runner = MistralRsBuilder::new(pipeline, scheduler_config, false, None);
    /// ```
    pub fn build_async(self) -> Result<std::sync::Arc<tokio::sync::Mutex<dyn crate::Pipeline + Send + Sync>>> {
        let pipeline = self.build()?;
        Ok(std::sync::Arc::new(tokio::sync::Mutex::new(pipeline)))
    }

    /// Resolve GGUF model source to local file paths.
    fn resolve_gguf_paths(&self) -> Result<Vec<PathBuf>> {
        match &self.source {
            ModelSource::LocalGguf(paths) => {
                // Verify paths exist
                for path in paths {
                    if !path.exists() {
                        bail!("GGUF file not found: {}", path.display());
                    }
                }
                Ok(paths.clone())
            }
            ModelSource::HuggingFaceGguf { repo_id, filenames, revision } => {
                if !self.silent {
                    info!("Downloading GGUF model from HuggingFace: {}", repo_id);
                }

                let api = Api::new().map_err(|e| anyhow!("Failed to create HF API: {}", e))?;
                let repo = match revision {
                    Some(rev) => api.repo(Repo::with_revision(
                        repo_id.clone(),
                        RepoType::Model,
                        rev.clone(),
                    )),
                    None => api.repo(Repo::new(repo_id.clone(), RepoType::Model)),
                };

                let mut paths = Vec::new();
                for filename in filenames {
                    let path = repo.get(filename).map_err(|e| {
                        anyhow!("Failed to download {}/{}: {}", repo_id, filename, e)
                    })?;
                    paths.push(path);
                }

                Ok(paths)
            }
            ModelSource::HuggingFaceSafetensors { .. } => {
                bail!("resolve_gguf_paths called for safetensors source")
            }
        }
    }

    /// Build pipeline for detected architecture.
    fn build_for_architecture(
        &self,
        arch: GGUFArchitecture,
        loader: &GgufLoader,
    ) -> Result<crate::pipeline::CausalLMPipeline> {
        use crate::pipeline::CausalLMPipeline;

        match arch {
            GGUFArchitecture::Llama => {
                // Check for MoE: Mixtral uses Llama architecture but with experts
                if loader.metadata().expert_count > 1 {
                    info!(
                        "Detected MoE model (expert_count={}), using Mixtral",
                        loader.metadata().expert_count
                    );
                    let pipeline = self.build_typed_pipeline::<Mixtral>(loader)?;
                    Ok(CausalLMPipeline::Mixtral(pipeline))
                } else {
                    let pipeline = self.build_typed_pipeline::<LlamaModel>(loader)?;
                    Ok(CausalLMPipeline::Llama(pipeline))
                }
            }
            GGUFArchitecture::Mistral3 => {
                let pipeline = self.build_typed_pipeline::<QMistral3>(loader)?;
                Ok(CausalLMPipeline::Mistral3(pipeline))
            }
            GGUFArchitecture::Phi2 => {
                let pipeline = self.build_typed_pipeline::<QPhi2>(loader)?;
                Ok(CausalLMPipeline::Phi2(pipeline))
            }
            GGUFArchitecture::Phi3 => {
                let pipeline = self.build_typed_pipeline::<QPhi3>(loader)?;
                Ok(CausalLMPipeline::Phi3(pipeline))
            }
            GGUFArchitecture::Starcoder2 => {
                let pipeline = self.build_typed_pipeline::<QStarcoder2>(loader)?;
                Ok(CausalLMPipeline::Starcoder2(pipeline))
            }
            GGUFArchitecture::Qwen2 => {
                let pipeline = self.build_typed_pipeline::<QQwen>(loader)?;
                Ok(CausalLMPipeline::Qwen2(pipeline))
            }
            GGUFArchitecture::Qwen3 => {
                let pipeline = self.build_typed_pipeline::<QQwen3>(loader)?;
                Ok(CausalLMPipeline::Qwen3(pipeline))
            }
            GGUFArchitecture::Qwen3MoE => {
                let pipeline = self.build_typed_pipeline::<QQwen3MoE>(loader)?;
                Ok(CausalLMPipeline::Qwen3MoE(pipeline))
            }
            arch => bail!(
                "Unsupported GGUF architecture `{arch}` for text pipeline. \
                 Supported: Llama, Mistral3, Phi2, Phi3, Starcoder2, Qwen2, Qwen3, Qwen3MoE"
            ),
        }
    }

    /// Build a typed pipeline for a specific model type.
    fn build_typed_pipeline<M: LanguageModel + FromGGUF + Send + Sync + 'static>(
        &self,
        loader: &GgufLoader,
    ) -> Result<crate::pipeline::TextPipeline<M>> {
        use crate::pipeline::chat_template::{ChatTemplate, ChatTemplateValue};
        use crate::pipeline::{GeneralMetadata, ModelKind, Modalities, QuantizationKind, SupportedModality};
        use either::Either;

        // Open files and parse content
        // Note: Use File directly (not BufReader) to match GgufDeviceMapLoaderInner's expected type
        let mut readers: Vec<File> = loader
            .paths
            .iter()
            .map(|p| File::open(p).map_err(|e| anyhow!("Failed to open {}: {}", p.display(), e)))
            .collect::<Result<Vec<_>>>()?;

        let mut reader_refs: Vec<&mut File> = readers.iter_mut().collect();
        let content = Content::from_readers(&mut reader_refs)?;

        // Extract model config from GGUF content (needed for paged attention)
        let content_config: ContentConfig = (&content).into();
        let arch = content.arch();

        // Get total layers from GGUF metadata
        let total_layers = content.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        // For pipeline parallelism, compute device map for only the loaded layers
        let (num_layers, layer_range_for_sizes) = match &self.layer_range {
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

        // Create device mapper, handling Auto→Map conversion
        let mut device_map_setting = self.device_map_setting.clone();

        if let DeviceMapSetting::Auto(params) = device_map_setting.clone() {
            let devices = device_map::get_all_similar_devices(&self.device)?;
            // Initial dtype
            let dtype = self.dtype;

            let model_inner = GgufDeviceMapLoaderInner {
                model: &content,
                arch,
            };

            // Get layer sizes, slicing to loaded range for pipeline parallelism
            let all_layer_sizes =
                model_inner.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let layer_sizes_in_bytes = match &layer_range_for_sizes {
                Some(range) => all_layer_sizes[range.clone()].to_vec(),
                None => all_layer_sizes,
            };

            // For pipeline parallelism, only include non-mapped tensors for stages that have them:
            // - First stage (layer 0): token_embd
            // - Last stage (final layer): output_norm + output/lm_head
            // - Middle stages: neither
            let non_mapped_size_in_bytes = match &self.layer_range {
                Some(range) => {
                    let is_first_stage = range.start == 0;
                    let is_last_stage = range.end >= total_layers;

                    let mut size = 0usize;

                    // First stage needs token embeddings
                    if is_first_stage {
                        if let Ok(t) = content.tensor_info("token_embd.weight") {
                            // Embeddings are dequantized to F32 at runtime
                            size += t.shape.elem_count() * DType::F32.size_in_bytes();
                        }
                    }

                    // Last stage needs output_norm and lm_head
                    if is_last_stage {
                        if let Ok(t) = content.tensor_info("output_norm.weight") {
                            size += t.shape.elem_count() * DType::F32.size_in_bytes();
                        }
                        // output.weight (lm_head) - may be tied to token_embd
                        if content.has_tensor("output.weight") {
                            if let Ok(t) = content.tensor_info("output.weight") {
                                size += t.shape.elem_count() / t.ggml_dtype.block_size() * t.ggml_dtype.type_size();
                            }
                        } else if let Ok(t) = content.tensor_info("token_embd.weight") {
                            // Tied embeddings - reuse token_embd for output
                            size += t.shape.elem_count() / t.ggml_dtype.block_size() * t.ggml_dtype.type_size();
                        }
                    }

                    debug!(
                        is_first_stage,
                        is_last_stage,
                        non_mapped_size_mib = size / (1024 * 1024),
                        "Pipeline stage non-mapped size (embed/lm_head)"
                    );
                    size
                }
                None => {
                    // No pipeline parallelism - include all non-mapped tensors
                    model_inner.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None)?
                }
            };
            let total_model_size_in_bytes =
                layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

            let new = model_inner.get_device_layers(
                "this is a dummy config!",
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                self.paged_attn_config.as_ref(),
            )?;
            device_map_setting = DeviceMapSetting::Map(new);
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &self.device {
            unsafe { dev.disable_event_tracking() };
        }

        let mapper = device_map_setting.into_mapper(num_layers, &self.device, self.topology.as_ref())?;

        // Check for CPU offloading with pipeline parallelism
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu {
            if let Some(ref range) = self.layer_range {
                bail!(
                    "Pipeline parallelism: layers {}..{} don't fit on available GPU(s). \
                     The coordinator assigned {} layers to this node, but they require CPU offloading. \
                     Either increase GPU memory, reduce model size, or adjust pipeline stage splits.",
                    range.start,
                    range.end.min(total_layers),
                    num_layers
                );
            }
            if self.paged_attn_config.is_some() {
                warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            }
        }

        // Load model via FromGGUF trait
        let model = M::from_gguf(
            content,
            &self.device,
            mapper,
            self.attention,
            self.dtype,
            self.layer_range.clone(),
        )
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        // Get tokenizer - prefer tok_model_id if specified, otherwise extract from GGUF
        let tokenizer = if let Some(ref tok_model_id) = self.tok_model_id {
            // Download tokenizer from separate HF repo
            if !self.silent {
                info!("Loading tokenizer from HuggingFace: {}", tok_model_id);
            }
            let api = Api::new().map_err(|e| anyhow!("Failed to create HF API: {}", e))?;
            let repo = api.repo(Repo::new(tok_model_id.clone(), RepoType::Model));
            let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                anyhow!("Failed to download tokenizer.json from {}: {}", tok_model_id, e)
            })?;
            let tokenizer = crate::utils::tokenizer::get_tokenizer(&tokenizer_path, None)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
            Arc::new(tokenizer)
        } else {
            // Extract tokenizer from GGUF
            loader.tokenizer()?
        };

        // Build LLG factory for grammar-constrained generation
        // build_llg_factory takes Tokenizer by value, so we clone the Arc's contents
        let llg_factory = build_llg_factory((*tokenizer).clone())
            .map_err(|e| anyhow!("Failed to build LLG factory: {}", e))?;

        // Get chat template (jinja_explicit > explicit string > embedded in GGUF)
        let mut chat_template = ChatTemplate::default();
        if let Some(ref jinja_path) = self.jinja_explicit {
            // Load from explicit Jinja file
            let template_content = std::fs::read_to_string(jinja_path)
                .map_err(|e| anyhow!("Failed to read Jinja template from {}: {}", jinja_path, e))?;
            chat_template.chat_template = Some(ChatTemplateValue(Either::Left(template_content)));
        } else if let Some(ref template) = self.chat_template {
            chat_template.chat_template = Some(ChatTemplateValue(Either::Left(template.clone())));
        } else if let Ok(Some(template)) = loader.chat_template_string() {
            chat_template.chat_template = Some(ChatTemplateValue(Either::Left(template)));
        }

        // Calculate EOS tokens for generation stopping
        // Note: BOS/EOS tokens are typically embedded in the tokenizer or chat template
        let eos_tokens = calculate_eos_tokens(&chat_template, None, &tokenizer);

        // Build model ID from source.
        // For HuggingFace models, prefer tok_model_id if specified (matches GGUFPipeline behavior).
        // This is important for pipeline parallelism where the model ID is used as the engine key.
        let model_id = self.tok_model_id.clone().unwrap_or_else(|| match &self.source {
            ModelSource::LocalGguf(paths) => paths
                .first()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            ModelSource::HuggingFaceGguf { repo_id, .. } => repo_id.clone(),
            ModelSource::HuggingFaceSafetensors { repo_id, .. } => repo_id.clone(),
        });

        // Build metadata
        let num_hidden_layers = model.num_layers();
        let max_seq_len = model.max_seq_len();

        // Setup paged attention if configured
        let (cache_config, cache_engine) = if let Some(ref paged_config) = self.paged_attn_config {
            // Adjust num_layers for pipeline parallelism
            let effective_num_layers = self.layer_range.as_ref()
                .map(|r| r.len())
                .unwrap_or(num_hidden_layers);

            let model_config = ModelConfigMetadata {
                max_seq_len,
                num_layers: effective_num_layers,
                hidden_size: content_config.hidden_size(),
                num_kv_heads: content_config.num_kv_heads(),
                num_attn_heads: content_config.num_attn_heads(),
                sliding_window: None,
                k_head_dim: content_config.k_head_dim(),
                v_head_dim: content_config.v_head_dim(),
            };

            info!(
                "Setting up PagedAttention: {} layers, {} KV heads, {} head dim",
                effective_num_layers,
                model_config.num_kv_heads,
                model_config.k_head_dim
            );

            // For now, use single device for layer mapping
            // TODO: Support multi-device paged attention
            let layer_devices: Vec<Option<Device>> = vec![Some(self.device.clone()); effective_num_layers];

            let config = calculate_cache_config(
                paged_config.mem_gpu,
                paged_config.block_size,
                self.dtype,
                paged_config.cache_type,
                &model_config,
                &self.device,
                &layer_devices,
                self.silent,
            )?;

            let engine = CacheEngine::new(
                &model_config,
                &config,
                self.dtype,
                &self.device,
                layer_devices,
            )?;

            (Some(config), Some(engine))
        } else {
            (None, None)
        };

        // Build model config metadata for GeneralMetadata
        let model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>> = Some(Arc::new(ModelConfigMetadata {
            max_seq_len,
            num_layers: num_hidden_layers,
            hidden_size: content_config.hidden_size(),
            num_kv_heads: content_config.num_kv_heads(),
            num_attn_heads: content_config.num_attn_heads(),
            sliding_window: None,
            k_head_dim: content_config.k_head_dim(),
            v_head_dim: content_config.v_head_dim(),
        }));

        let metadata = Arc::new(GeneralMetadata {
            max_seq_len,
            llg_factory: Some(llg_factory),
            no_kv_cache: self.no_kv_cache,
            no_prefix_cache: false,
            num_hidden_layers,
            eos_tok: eos_tokens,
            kind: ModelKind::GgufQuantized {
                quant: QuantizationKind::Gguf,
            },
            is_xlora: false,
            activation_dtype: self.dtype,
            sliding_window: None,
            cache_config,
            cache_engine,
            model_metadata,
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
        });

        // Create pipeline mapper (model consumed the original mapper)
        let pipeline_mapper: Box<dyn DeviceMapper + Send + Sync> =
            Box::new(crate::device_map::SingleDeviceMapper::new(self.device.clone()));

        Ok(crate::pipeline::TextPipeline::new(
            model,
            tokenizer,
            Arc::new(chat_template),
            model_id,
            pipeline_mapper,
            metadata,
        ))
    }
}

// =============================================================================
// CausalLMLoader - Implements Loader trait for full system integration
// =============================================================================

use crate::pipeline::{Loader, ModelKind, ModelPaths, QuantizationKind};
use crate::TryIntoDType;
use tokio::sync::Mutex;

/// Loader implementing the `Loader` trait for GGUF causal language models.
///
/// This provides full compatibility with the existing loader system, allowing
/// `CausalLMPipeline` to be used wherever `GGUFLoader` was used before.
///
/// # Example
///
/// ```ignore
/// // Create loader (similar to GGUFLoaderBuilder)
/// let loader = CausalLMLoader::new(
///     "model-repo",
///     vec!["model.gguf".to_string()],
///     None, // chat_template
///     false, // no_kv_cache
/// );
///
/// // Use with existing infrastructure
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     &ModelDType::Auto,
///     &Device::Cpu,
///     false,
///     DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
///     None,
///     None,
/// )?;
/// ```
pub struct CausalLMLoader {
    /// HuggingFace model ID or local path identifier
    model_id: String,
    /// GGUF filenames (for HF) or full paths (for local)
    filenames: Vec<String>,
    /// Optional explicit chat template
    chat_template: Option<String>,
    /// Whether to disable KV cache
    no_kv_cache: bool,
    /// Optional layer range for pipeline parallelism
    layer_range: Option<Range<usize>>,
    /// Optional tokenizer model ID (for sourcing tokenizer from different HF repo)
    tok_model_id: Option<String>,
    /// Optional topology for device mapping
    topology: Option<crate::Topology>,
    /// Optional explicit Jinja template file path
    jinja_explicit: Option<String>,
}

impl CausalLMLoader {
    /// Create a new loader for GGUF models.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace repo ID or local path identifier
    /// * `filenames` - GGUF filename(s) within the repository
    /// * `chat_template` - Optional explicit chat template
    /// * `no_kv_cache` - Whether to disable KV cache
    pub fn new(
        model_id: impl ToString,
        filenames: Vec<String>,
        chat_template: Option<String>,
        no_kv_cache: bool,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            filenames,
            chat_template,
            no_kv_cache,
            layer_range: None,
            tok_model_id: None,
            topology: None,
            jinja_explicit: None,
        }
    }

    /// Set the layer range for pipeline parallelism.
    pub fn with_layer_range(mut self, range: Range<usize>) -> Self {
        self.layer_range = Some(range);
        self
    }

    /// Set the tokenizer model ID for sourcing tokenizer from a different HF repo.
    pub fn with_tok_model_id(mut self, tok_model_id: String) -> Self {
        self.tok_model_id = Some(tok_model_id);
        self
    }

    /// Set the device mapping topology.
    pub fn with_topology(mut self, topology: crate::Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Set an explicit Jinja template file path.
    pub fn with_jinja_explicit(mut self, jinja_path: String) -> Self {
        self.jinja_explicit = Some(jinja_path);
        self
    }
}

impl Loader for CausalLMLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn crate::Pipeline + Send + Sync>>> {
        // Build using CausalLMLoaderBuilder
        let mut builder = CausalLMLoaderBuilder::from_hf_gguf(
            &self.model_id,
            &self.filenames.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )
        .with_device(device.clone())
        .with_token_source(token_source)
        .with_device_map(mapper)
        .with_no_kv_cache(self.no_kv_cache);

        if let Some(rev) = revision {
            builder = builder.with_revision(rev);
        }

        if let Some(ref template) = self.chat_template {
            builder = builder.with_chat_template(template.clone());
        }

        if let Some(ref range) = self.layer_range {
            builder = builder.with_layer_range(range.clone());
        }

        if let Some(ref tok_model_id) = self.tok_model_id {
            builder = builder.with_tok_model_id(tok_model_id.clone());
        }

        if let Some(ref topology) = self.topology {
            builder = builder.with_topology(topology.clone());
        }

        if let Some(ref jinja_path) = self.jinja_explicit {
            builder = builder.with_jinja_explicit(jinja_path.clone());
        }

        if let Some(config) = paged_attn_config {
            builder = builder.with_paged_attention(config);
        }

        if silent {
            builder = builder.silent();
        }

        // Get dtype from TryIntoDType
        let devices = vec![device];
        let actual_dtype = dtype.try_into_dtype(&devices)?;
        builder = builder.with_dtype(actual_dtype);

        builder.build_async()
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments, clippy::borrowed_box)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn crate::Pipeline + Send + Sync>>> {
        // Get actual file paths
        let weight_paths = paths.get_weight_filenames();

        // Build using CausalLMLoaderBuilder
        let mut builder = CausalLMLoaderBuilder::from_gguf_paths(weight_paths)
            .with_device(device.clone())
            .with_device_map(mapper)
            .with_no_kv_cache(self.no_kv_cache);

        if let Some(ref template) = self.chat_template {
            builder = builder.with_chat_template(template.clone());
        }

        if let Some(ref range) = self.layer_range {
            builder = builder.with_layer_range(range.clone());
        }

        if let Some(ref tok_model_id) = self.tok_model_id {
            builder = builder.with_tok_model_id(tok_model_id.clone());
        }

        if let Some(ref topology) = self.topology {
            builder = builder.with_topology(topology.clone());
        }

        if let Some(ref jinja_path) = self.jinja_explicit {
            builder = builder.with_jinja_explicit(jinja_path.clone());
        }

        if let Some(config) = paged_attn_config {
            builder = builder.with_paged_attention(config);
        }

        if silent {
            builder = builder.silent();
        }

        // Get dtype from TryIntoDType
        let devices = vec![device];
        let actual_dtype = dtype.try_into_dtype(&devices)?;
        builder = builder.with_dtype(actual_dtype);

        builder.build_async()
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_loader_requires_paths() {
        let empty: [&str; 0] = [];
        let result = GgufLoader::open(&empty);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No GGUF files"));
    }
}
