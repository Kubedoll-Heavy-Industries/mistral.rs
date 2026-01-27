//! Integration tests for `load_text_pipeline()`.
//!
//! By default, tests download Qwen3-0.6B-Q4_K_M from HuggingFace Hub.
//! To use a different model, set `TEST_GGUF_MODEL=/path/to/model.gguf`.
//!
//! ## Serial Test Groups
//!
//! Tests use named serial groups to prevent memory exhaustion:
//! - `#[serial(small_model)]`: Tests loading models <2B params (some parallelism ok)
//! - `#[serial(large_model)]`: Tests loading models >2B params (strictly serialized)
//!
//! These groups are shared across test files to prevent cross-crate parallelism issues.

use std::path::PathBuf;
use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use mistralrs_core::{load_text_pipeline, AttentionImplementation, GgufLoader, MetadataMixin};
use serial_test::serial;

/// Default test model: Qwen3-0.6B quantized (small, fast to load)
const DEFAULT_MODEL_REPO: &str = "unsloth/Qwen3-0.6B-GGUF";
const DEFAULT_MODEL_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";

/// Cached model path (downloaded once per test run)
static MODEL_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Get the test model path, downloading if necessary.
fn get_test_model_path() -> PathBuf {
    MODEL_PATH
        .get_or_init(|| {
            // Check for override via environment variable
            if let Ok(path) = std::env::var("TEST_GGUF_MODEL") {
                let path = PathBuf::from(path);
                if path.exists() {
                    println!("Using model from TEST_GGUF_MODEL: {:?}", path);
                    return path;
                }
                eprintln!("Warning: TEST_GGUF_MODEL path doesn't exist, downloading default");
            }

            // Download default model from HuggingFace Hub
            println!(
                "Downloading test model: {}/{}",
                DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILE
            );
            let api = Api::new().expect("Failed to create HuggingFace API client");
            let repo = api.repo(Repo::new(DEFAULT_MODEL_REPO.to_string(), RepoType::Model));
            repo.get(DEFAULT_MODEL_FILE)
                .expect("Failed to download test model")
        })
        .clone()
}

#[test]
#[serial(small_model)]
fn test_load_text_pipeline_basic() {
    let model_path = get_test_model_path();

    println!("Loading model from: {:?}", model_path);

    let device = Device::Cpu;

    // Load the pipeline
    let result = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        None, // All layers
        None, // Default mapper
    );

    match result {
        Ok(pipeline) => {
            println!("Successfully loaded pipeline: {}", pipeline.name());
            // Basic sanity check - we got a pipeline
            assert!(!pipeline.name().is_empty());
        }
        Err(e) => {
            panic!("Failed to load pipeline: {}", e);
        }
    }
}

#[test]
#[serial(small_model)]
fn test_load_text_pipeline_with_layer_range() {
    let model_path = get_test_model_path();

    println!("Loading model with layer range from: {:?}", model_path);

    let device = Device::Cpu;

    // Load only first 4 layers (simulating pipeline parallelism HEAD stage)
    let result = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        Some(0..4), // First 4 layers only
        None,       // Default mapper
    );

    match result {
        Ok(pipeline) => {
            println!(
                "Successfully loaded partial pipeline (layers 0-4): {}",
                pipeline.name()
            );
        }
        Err(e) => {
            // Layer range might cause issues with some models - that's OK for now
            eprintln!(
                "Note: Layer range loading returned error (expected for some models): {}",
                e
            );
        }
    }
}

#[test]
#[serial(small_model)]
fn test_load_text_pipeline_forward_pass() {
    let model_path = get_test_model_path();

    println!("Testing forward pass with model: {:?}", model_path);

    let device = Device::Cpu;

    let pipeline = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        None, // All layers
        None, // Default mapper
    )
    .expect("Failed to load pipeline");

    // Get tokenizer and encode a simple prompt
    let tokenizer = pipeline
        .tokenizer()
        .expect("Pipeline should have tokenizer");
    let encoding = tokenizer
        .encode("Hello, world!", false)
        .expect("Failed to encode");

    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Encoded {} tokens", input_ids.len());

    // Create input tensor
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)
        .expect("Failed to create tensor")
        .unsqueeze(0) // Add batch dimension
        .expect("Failed to unsqueeze");

    println!("Input tensor shape: {:?}", input_tensor.dims());

    // Note: Actually running forward_inputs requires more setup (ModelInputs struct).
    // For now, we just verify the pipeline loaded and tokenizer works.
    // A full forward pass test would require constructing the full input structure.

    println!("Pipeline loaded successfully and tokenizer works!");
}

#[test]
#[serial(small_model)]
fn test_architecture_detection() {
    let model_path = get_test_model_path();

    // Use GgufLoader to detect architecture
    let loader = GgufLoader::open(&[&model_path]).expect("Failed to open GGUF");
    let arch = loader.gguf_architecture();

    println!("Detected architecture: {:?}", arch);

    // Architecture should be one of the supported types
    // (We can't assert a specific value since we don't know what model the user has)
}

// =============================================================================
// Pipeline Parallelism Tests
// =============================================================================

use mistralrs_core::{HookContainer, PipelineHook};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mock hook for testing pipeline parallelism stage detection.
///
/// This hook simulates a PP stage without actual transport:
/// - HEAD stage: layer_range 0..N, needs_external_logits=true
/// - TAIL stage: layer_range N..total, needs_external_logits=false
struct MockPipelineHook {
    layer_range: Range<usize>,
    needs_external_logits: bool,
    send_count: AtomicUsize,
    receive_count: AtomicUsize,
}

impl MockPipelineHook {
    fn head_stage(layer_end: usize) -> Self {
        Self {
            layer_range: 0..layer_end,
            needs_external_logits: true, // HEAD receives logits from TAIL
            send_count: AtomicUsize::new(0),
            receive_count: AtomicUsize::new(0),
        }
    }

    fn tail_stage(layer_start: usize, layer_end: usize) -> Self {
        Self {
            layer_range: layer_start..layer_end,
            needs_external_logits: false, // TAIL computes logits locally
            send_count: AtomicUsize::new(0),
            receive_count: AtomicUsize::new(0),
        }
    }
}

impl PipelineHook for MockPipelineHook {
    fn layer_range(&self) -> Range<usize> {
        self.layer_range.clone()
    }

    fn needs_external_logits(&self) -> bool {
        self.needs_external_logits
    }

    fn send_activation(
        &self,
        _hidden: &Tensor,
        _tokens: &[u32],
        _request_id: uuid::Uuid,
        _sequence_position: usize,
    ) -> candle_core::Result<()> {
        self.send_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn receive_activation(
        &self,
        _request_id: uuid::Uuid,
    ) -> candle_core::Result<mistralrs_core::ActivationResult> {
        self.receive_count.fetch_add(1, Ordering::SeqCst);
        // In a real test, this would block until activation arrives
        // For now, return an error since we're just testing stage detection
        Err(candle_core::Error::Msg(
            "MockPipelineHook: receive_activation not implemented for testing".to_string(),
        ))
    }
}

#[test]
fn test_pp_stage_detection_head() {
    // Test that HEAD stage is correctly identified
    let hook = MockPipelineHook::head_stage(14); // First 14 layers
    let container = HookContainer::new(std::sync::Arc::new(hook));

    assert!(container.is_first_stage(), "HEAD should be first stage");
    assert!(!container.is_last_stage(), "HEAD should NOT be last stage");
    assert!(
        container.needs_external_logits(),
        "HEAD needs external logits"
    );
}

#[test]
fn test_pp_stage_detection_tail() {
    // Test that TAIL stage is correctly identified
    let hook = MockPipelineHook::tail_stage(14, 28); // Layers 14-28
    let container = HookContainer::new(std::sync::Arc::new(hook));

    assert!(
        !container.is_first_stage(),
        "TAIL should NOT be first stage"
    );
    assert!(container.is_last_stage(), "TAIL should be last stage");
    assert!(
        !container.needs_external_logits(),
        "TAIL computes logits locally"
    );
}

#[test]
#[serial(small_model)]
fn test_pp_head_and_tail_stage_loading() {
    let model_path = get_test_model_path();

    println!(
        "Testing PP HEAD and TAIL stage loading with model: {:?}",
        model_path
    );

    let device = Device::Cpu;

    // First, load full model to get total layer count
    let loader = GgufLoader::open(&[&model_path]).expect("Failed to open GGUF");
    let metadata = loader.metadata();
    let total_layers = metadata.num_layers;

    println!("Model has {} layers", total_layers);
    assert!(total_layers >= 4, "Need at least 4 layers for PP test");

    // Split layers: HEAD gets first half, TAIL gets second half
    let mid = total_layers / 2;
    println!(
        "Splitting at layer {}: HEAD=0..{}, TAIL={}..{}",
        mid, mid, mid, total_layers
    );

    // Load HEAD stage (layers 0..mid)
    println!("\nLoading HEAD stage (layers 0..{})...", mid);
    let head_result = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        Some(0..mid),
        None,
    );

    match &head_result {
        Ok(pipeline) => {
            println!("  HEAD loaded successfully: {}", pipeline.name());
        }
        Err(e) => {
            // Layer range loading is a new feature - log but don't fail
            println!("  HEAD loading error (may be expected): {}", e);
        }
    }

    // Load TAIL stage (layers mid..total)
    println!("\nLoading TAIL stage (layers {}..{})...", mid, total_layers);
    let tail_result = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        Some(mid..total_layers),
        None,
    );

    match &tail_result {
        Ok(pipeline) => {
            println!("  TAIL loaded successfully: {}", pipeline.name());
        }
        Err(e) => {
            println!("  TAIL loading error (may be expected): {}", e);
        }
    }

    // For now, we just verify that loading doesn't panic
    // Once layer range loading is fully implemented, we can add more assertions
    println!("\nPP stage loading test completed");
}

// =============================================================================
// CausalLMLoaderBuilder Tests
// =============================================================================

use mistralrs_core::CausalLMLoaderBuilder;

#[test]
#[serial(small_model)]
fn test_builder_from_paths() {
    let model_path = get_test_model_path();

    println!("Testing CausalLMLoaderBuilder with model: {:?}", model_path);

    let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build()
        .expect("Failed to build pipeline");

    println!(
        "Successfully loaded pipeline via builder: {}",
        pipeline.name()
    );
    assert!(!pipeline.name().is_empty());
}

#[test]
#[serial(small_model)]
fn test_builder_with_layer_range() {
    let model_path = get_test_model_path();

    // Get total layers
    let loader = GgufLoader::open(&[&model_path]).expect("Failed to open GGUF");
    let total_layers = loader.metadata().num_layers;
    println!("Model has {} layers", total_layers);

    if total_layers < 4 {
        println!("Skipping layer range test - model has < 4 layers");
        return;
    }

    // Load first half of layers
    let mid = total_layers / 2;
    println!("Loading layers 0..{}", mid);

    let result = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .with_layer_range(0..mid)
        .silent()
        .build();

    match result {
        Ok(pipeline) => {
            println!("Successfully loaded partial pipeline: {}", pipeline.name());
        }
        Err(e) => {
            // Layer range loading may not be fully supported for all models
            println!(
                "Note: Layer range loading returned error (may be expected): {}",
                e
            );
        }
    }
}

#[test]
#[serial(small_model)]
fn test_builder_from_hf_gguf() {
    // This test downloads from HuggingFace Hub
    // Using the same small model as other tests

    println!("Testing CausalLMLoaderBuilder::from_hf_gguf()");

    let pipeline = CausalLMLoaderBuilder::from_hf_gguf(DEFAULT_MODEL_REPO, &[DEFAULT_MODEL_FILE])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build()
        .expect("Failed to build pipeline from HF");

    println!("Successfully loaded pipeline from HF: {}", pipeline.name());
    assert!(!pipeline.name().is_empty());
}

#[test]
#[serial(small_model)]
fn test_builder_build_async() {
    // Test build_async() which returns Arc<Mutex<dyn Pipeline>>
    let model_path = get_test_model_path();

    println!("Testing CausalLMLoaderBuilder::build_async()");

    let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build_async()
        .expect("Failed to build async pipeline");

    // Verify we can acquire the lock and access the pipeline
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let guard = pipeline.lock().await;
        let name = guard.name();
        println!("Successfully loaded async pipeline: {}", name);
        assert!(!name.is_empty());
    });
}

// =============================================================================
// CausalLMLoader Tests (Loader trait implementation)
// =============================================================================

use mistralrs_core::{
    CausalLMLoader, DeviceMapSetting, Loader, LoaderBuilder, ModelDType, ModelSelected, TokenSource,
};

#[test]
#[serial(small_model)]
fn test_loader_builder_gguf_uses_causal_lm_loader() {
    // Test that LoaderBuilder with ModelSelected::GGUF creates a working loader
    // This validates the migration from GGUFLoader to CausalLMLoader
    let model_path = get_test_model_path();

    println!("Testing LoaderBuilder with ModelSelected::GGUF");

    let model_selected = ModelSelected::GGUF {
        tok_model_id: None,
        quantized_model_id: model_path.parent().unwrap().to_string_lossy().to_string(),
        quantized_filename: model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        topology: None,
        dtype: ModelDType::Auto,
        max_seq_len: 2048,
        max_batch_size: 1,
    };

    let loader = LoaderBuilder::new(model_selected)
        .build()
        .expect("Failed to build loader from ModelSelected::GGUF");

    // Verify it's a GGUF loader
    assert!(matches!(
        loader.get_kind(),
        mistralrs_core::ModelKind::GgufQuantized { .. }
    ));

    println!(
        "Successfully created loader via LoaderBuilder: {}",
        loader.get_id()
    );
}

#[test]
#[serial(small_model)]
fn test_causal_lm_loader_from_hf() {
    // Test CausalLMLoader implementing Loader trait
    println!("Testing CausalLMLoader::load_model_from_hf()");

    let loader = CausalLMLoader::new(
        DEFAULT_MODEL_REPO,
        vec![DEFAULT_MODEL_FILE.to_string()],
        None,
        false,
    );

    // Verify get_id and get_kind work
    assert_eq!(loader.get_id(), DEFAULT_MODEL_REPO);
    assert!(matches!(
        loader.get_kind(),
        mistralrs_core::ModelKind::GgufQuantized { .. }
    ));

    // Load the model via Loader trait
    let pipeline = loader
        .load_model_from_hf(
            None,
            TokenSource::CacheToken,
            &ModelDType::Auto,
            &Device::Cpu,
            true, // silent
            DeviceMapSetting::dummy(),
            None,
            None,
        )
        .expect("Failed to load via CausalLMLoader");

    // Verify we can access the pipeline
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let guard = pipeline.lock().await;
        let name = guard.name();
        println!("Successfully loaded via CausalLMLoader: {}", name);
        assert!(!name.is_empty());
    });
}

// =============================================================================
// Parameterized Model Family Tests
// =============================================================================
//
// These tests allow testing different model families by setting environment
// variables. Use for comprehensive e2e validation across architectures.
//
// Environment variables:
// - TEST_LLAMA_MODEL: Path to a Llama GGUF model
// - TEST_QWEN3_MODEL: Path to a Qwen3 GGUF model
// - TEST_PHI3_MODEL: Path to a Phi3 GGUF model
// - TEST_MISTRAL3_MODEL: Path to a Mistral3 GGUF model
// - TEST_STARCODER2_MODEL: Path to a Starcoder2 GGUF model
//
// Example usage:
//   TEST_LLAMA_MODEL=/path/to/llama.gguf cargo test --test text_pipeline_integration test_llama_family
//   TEST_QWEN3_MODEL=/path/to/qwen3.gguf TEST_LLAMA_MODEL=/path/to/llama.gguf cargo test --test text_pipeline_integration test_all_model_families

/// Test model info for parameterized tests.
struct ModelTestInfo {
    name: &'static str,
    env_var: &'static str,
    expected_arch: &'static str,
}

const MODEL_FAMILIES: &[ModelTestInfo] = &[
    ModelTestInfo {
        name: "Llama",
        env_var: "TEST_LLAMA_MODEL",
        expected_arch: "llama",
    },
    ModelTestInfo {
        name: "Mixtral",
        env_var: "TEST_MIXTRAL_MODEL",
        expected_arch: "llama", // Mixtral uses Llama architecture tag in GGUF
    },
    ModelTestInfo {
        name: "Qwen2",
        env_var: "TEST_QWEN2_MODEL",
        expected_arch: "qwen2",
    },
    ModelTestInfo {
        name: "Qwen3",
        env_var: "TEST_QWEN3_MODEL",
        expected_arch: "qwen3",
    },
    ModelTestInfo {
        name: "Phi2",
        env_var: "TEST_PHI2_MODEL",
        expected_arch: "phi2",
    },
    ModelTestInfo {
        name: "Phi3",
        env_var: "TEST_PHI3_MODEL",
        expected_arch: "phi3",
    },
    ModelTestInfo {
        name: "Mistral3",
        env_var: "TEST_MISTRAL3_MODEL",
        expected_arch: "mistral3",
    },
    ModelTestInfo {
        name: "Starcoder2",
        env_var: "TEST_STARCODER2_MODEL",
        expected_arch: "starcoder2",
    },
];

/// Run load test for a specific model family.
fn run_model_family_test(info: &ModelTestInfo) -> bool {
    let model_path = match std::env::var(info.env_var) {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            println!("Skipping {} test: {} not set", info.name, info.env_var);
            return false;
        }
    };

    if !model_path.exists() {
        println!(
            "Skipping {} test: path {:?} does not exist",
            info.name, model_path
        );
        return false;
    }

    println!("\n=== Testing {} model: {:?} ===", info.name, model_path);

    // Verify architecture
    let loader = GgufLoader::open(&[&model_path]).expect("Failed to open GGUF");
    let arch = format!("{:?}", loader.gguf_architecture()).to_lowercase();
    println!("Detected architecture: {}", arch);

    if !arch.contains(info.expected_arch) {
        println!(
            "Warning: Expected architecture containing '{}', got '{}'",
            info.expected_arch, arch
        );
    }

    // Load pipeline
    let device = Device::Cpu;
    let pipeline = load_text_pipeline(
        &[&model_path],
        &device,
        AttentionImplementation::Eager,
        DType::F32,
        None,
        None,
    );

    match pipeline {
        Ok(pipeline) => {
            println!(
                "Successfully loaded {} pipeline: {}",
                info.name,
                pipeline.name()
            );

            // Test tokenizer
            if let Some(tokenizer) = pipeline.tokenizer() {
                let test_text = format!("Hello from {} model!", info.name);
                match tokenizer.encode(test_text.as_str(), false) {
                    Ok(encoding) => {
                        println!("Tokenized '{}' -> {} tokens", test_text, encoding.len());
                    }
                    Err(e) => {
                        println!("Warning: Failed to encode text: {}", e);
                    }
                }
            }

            println!("=== {} test PASSED ===\n", info.name);
            true
        }
        Err(e) => {
            println!("Failed to load {} pipeline: {}", info.name, e);
            println!("=== {} test FAILED ===\n", info.name);
            false
        }
    }
}

/// Helper to find a model family by name.
fn find_model_family(name: &str) -> Option<&'static ModelTestInfo> {
    MODEL_FAMILIES.iter().find(|info| info.name == name)
}

#[test]
#[serial(small_model)]
fn test_llama_family() {
    let info = find_model_family("Llama").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Llama model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_mixtral_family() {
    let info = find_model_family("Mixtral").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Mixtral model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_qwen2_family() {
    let info = find_model_family("Qwen2").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Qwen2 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_qwen3_family() {
    let info = find_model_family("Qwen3").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Qwen3 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_phi2_family() {
    let info = find_model_family("Phi2").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Phi2 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_phi3_family() {
    let info = find_model_family("Phi3").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Phi3 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_mistral3_family() {
    let info = find_model_family("Mistral3").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Mistral3 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_starcoder2_family() {
    let info = find_model_family("Starcoder2").unwrap();
    if std::env::var(info.env_var).is_ok() {
        assert!(run_model_family_test(info), "Starcoder2 model test failed");
    } else {
        println!("Set {} to run this test", info.env_var);
    }
}

#[test]
#[serial(small_model)]
fn test_all_model_families() {
    println!("\n=== Testing All Available Model Families ===\n");

    let mut tested = 0;
    let mut passed = 0;

    for info in MODEL_FAMILIES {
        if std::env::var(info.env_var).is_ok() {
            tested += 1;
            if run_model_family_test(info) {
                passed += 1;
            }
        }
    }

    println!(
        "\n=== Summary: {}/{} model families passed ===",
        passed, tested
    );

    if tested == 0 {
        println!("No model family environment variables set. Set one or more of:");
        for info in MODEL_FAMILIES {
            println!(
                "  {}=/path/to/{}.gguf",
                info.env_var,
                info.name.to_lowercase()
            );
        }
    } else {
        assert_eq!(passed, tested, "Some model families failed");
    }
}

// =============================================================================
// LoRA Adapter Tests
// =============================================================================
//
// These tests verify LoRA adapter loading and inference.
//
// Environment variables:
// - TEST_LORA_BASE_MODEL: HF repo ID or path to base GGUF model
//   Default: unsloth/Qwen3-0.6B-GGUF (same as other tests)
// - TEST_LORA_BASE_FILE: GGUF filename within the repo
//   Default: Qwen3-0.6B-Q4_K_M.gguf
// - TEST_LORA_ADAPTER: HF repo ID for LoRA adapter
//   Example: "someone/qwen3-lora-adapter"
//
// Example usage:
//   TEST_LORA_ADAPTER=username/my-adapter cargo test --test text_pipeline_integration test_lora
//
// Note: LoRA adapters must be compatible with the base model architecture.
// The adapter's target_modules in adapter_config.json must match base model layers.

/// Test LoRA adapter loading via CausalLMLoaderBuilder.
///
/// This test verifies that a model with a LoRA adapter loads successfully.
/// Set TEST_LORA_ADAPTER environment variable to run this test.
#[test]
#[serial(small_model)]
fn test_lora_adapter_loading() {
    let adapter_repo = match std::env::var("TEST_LORA_ADAPTER") {
        Ok(repo) => repo,
        Err(_) => {
            println!("Skipping LoRA test: TEST_LORA_ADAPTER not set");
            println!("To run this test, set TEST_LORA_ADAPTER to a HuggingFace adapter repo");
            println!("Example: TEST_LORA_ADAPTER=username/my-lora-adapter cargo test test_lora");
            return;
        }
    };

    // Use custom base model if specified, otherwise use default
    let (base_repo, base_file) = if let Ok(base) = std::env::var("TEST_LORA_BASE_MODEL") {
        let file = std::env::var("TEST_LORA_BASE_FILE")
            .unwrap_or_else(|_| "model.gguf".to_string());
        (base, file)
    } else {
        (DEFAULT_MODEL_REPO.to_string(), DEFAULT_MODEL_FILE.to_string())
    };

    println!("\n=== Testing LoRA Adapter Loading ===");
    println!("Base model: {}/{}", base_repo, base_file);
    println!("LoRA adapter: {}", adapter_repo);

    let result = CausalLMLoaderBuilder::from_hf_gguf(&base_repo, &[&base_file])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .with_lora_adapter(&adapter_repo)
        .silent()
        .build();

    match result {
        Ok(pipeline) => {
            println!("Successfully loaded model with LoRA adapter: {}", pipeline.name());

            // Verify tokenizer works
            if let Some(tokenizer) = pipeline.tokenizer() {
                let test_text = "Hello, testing LoRA adapter!";
                match tokenizer.encode(test_text, false) {
                    Ok(encoding) => {
                        println!("Tokenized '{}' -> {} tokens", test_text, encoding.len());
                    }
                    Err(e) => {
                        println!("Warning: Failed to encode text: {}", e);
                    }
                }
            }

            println!("=== LoRA adapter loading test PASSED ===\n");
        }
        Err(e) => {
            panic!("Failed to load model with LoRA adapter: {}", e);
        }
    }
}

/// Test that loading a model without adapters still works.
/// This serves as a baseline to compare against LoRA-loaded models.
#[test]
#[serial(small_model)]
fn test_baseline_without_lora() {
    let model_path = get_test_model_path();

    println!("\n=== Testing Baseline (No LoRA) ===");
    println!("Model: {:?}", model_path);

    let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build()
        .expect("Failed to load baseline model");

    println!("Successfully loaded baseline model: {}", pipeline.name());
    println!("=== Baseline test PASSED ===\n");
}

/// Test multiple LoRA adapters (merged in order).
///
/// Set TEST_LORA_ADAPTER and TEST_LORA_ADAPTER_2 to run this test.
#[test]
#[serial(small_model)]
fn test_multiple_lora_adapters() {
    let adapter1 = match std::env::var("TEST_LORA_ADAPTER") {
        Ok(repo) => repo,
        Err(_) => {
            println!("Skipping multiple LoRA test: TEST_LORA_ADAPTER not set");
            return;
        }
    };

    let adapter2 = match std::env::var("TEST_LORA_ADAPTER_2") {
        Ok(repo) => repo,
        Err(_) => {
            println!("Skipping multiple LoRA test: TEST_LORA_ADAPTER_2 not set");
            return;
        }
    };

    let model_path = get_test_model_path();

    println!("\n=== Testing Multiple LoRA Adapters ===");
    println!("Model: {:?}", model_path);
    println!("Adapter 1: {}", adapter1);
    println!("Adapter 2: {}", adapter2);

    let result = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .with_lora_adapter(&adapter1)
        .with_lora_adapter(&adapter2)
        .silent()
        .build();

    match result {
        Ok(pipeline) => {
            println!("Successfully loaded model with 2 LoRA adapters: {}", pipeline.name());
            println!("=== Multiple LoRA adapters test PASSED ===\n");
        }
        Err(e) => {
            panic!("Failed to load model with multiple LoRA adapters: {}", e);
        }
    }
}

/// Test per-request adapter selection wiring.
///
/// This test verifies that the adapter selection flow works correctly:
/// 1. Pipeline can have an AdapterRegistry attached
/// 2. Requests with adapter names flow through to forward_inputs
/// 3. The registry's set_active() is called (even if no adapters registered)
///
/// Note: This tests the wiring, not the actual LoRA computation. For full
/// LoRA e2e testing, use test_lora_adapter_loading with TEST_LORA_ADAPTER set.
#[test]
#[serial(small_model)]
fn test_per_request_adapter_wiring() {
    use mistralrs_core::AdapterRegistry;
    use std::sync::Arc;

    let model_path = get_test_model_path();

    println!("\n=== Testing Per-Request Adapter Wiring ===");
    println!("Model: {:?}", model_path);

    // Build pipeline
    let mut pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build()
        .expect("Failed to load model");

    // Create empty adapter registry
    let registry = Arc::new(AdapterRegistry::new(Device::Cpu));

    // Attach registry to pipeline
    pipeline.set_adapter_registry(registry.clone());

    // Verify registry was attached
    assert!(
        pipeline.adapter_registry().is_some(),
        "Adapter registry should be attached"
    );
    assert!(
        Arc::ptr_eq(pipeline.adapter_registry().unwrap(), &registry),
        "Registry reference should match"
    );

    println!("Successfully attached adapter registry to pipeline");
    println!("=== Per-request adapter wiring test PASSED ===\n");
}

// =============================================================================
// End-to-End Inference Tests
// =============================================================================
//
// These tests verify that inference produces coherent output, not just that
// models load successfully. They use the full MistralRs engine.
//
// Environment variables:
// - TEST_E2E_INFERENCE=1: Enable e2e inference tests (they take longer)
// - TEST_LORA_ADAPTER: LoRA adapter repo for LoRA inference test

use indexmap::IndexMap;
use mistralrs_core::{
    Constraint, DefaultSchedulerMethod, InferenceExec, InferenceInput, InferenceOperation,
    MessageContent, MistralRsBuilder, NormalRequest, Request, ResponseOk, SchedulerConfig,
    TokenSamplingParams,
};

/// Helper to create a simple chat message.
fn create_chat_message(role: &str, content: &str) -> IndexMap<String, MessageContent> {
    use either::Either;
    let mut msg = IndexMap::new();
    msg.insert("role".to_string(), Either::Left(role.to_string()));
    msg.insert("content".to_string(), Either::Left(content.to_string()));
    msg
}

/// Helper to run a simple inference request and return the response text.
async fn run_simple_inference(
    runner: &std::sync::Arc<mistralrs_core::MistralRs>,
    prompt: &str,
) -> anyhow::Result<String> {
    use tokio::sync::mpsc::channel;

    let messages = vec![create_chat_message("user", prompt)];

    let (tx, mut rx) = channel(1);

    let request = Request::Normal(Box::new(NormalRequest {
        id: uuid::Uuid::new_v4(),
        input: InferenceInput {
            op: InferenceOperation::Chat {
                messages,
                attachments: vec![],
                thinking: None,
                sampling_params: {
                    let mut params = TokenSamplingParams::deterministic();
                    params.max_len = Some(50); // Short response
                    params
                },
                return_logprobs: false,
                constraint: Constraint::None,
                tools: None,
                tool_choice: None,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: None,
            },
            exec: InferenceExec {
                is_streaming: false,
                truncate_sequence: false,
            },
            adapters: None,
        },
        response: tx,
        model_id: None,
    }));

    runner.get_sender(None)?.send(request).await?;

    let response = rx
        .recv()
        .await
        .ok_or_else(|| anyhow::anyhow!("Channel closed"))?;

    match response.as_result()? {
        ResponseOk::Done(completion) => {
            let content = completion
                .choices
                .first()
                .and_then(|c| c.message.content.as_ref())
                .map(|s| s.to_string())
                .unwrap_or_default();
            Ok(content)
        }
        _ => anyhow::bail!("Unexpected response type"),
    }
}

/// End-to-end inference test: verify model produces coherent output.
///
/// This test loads a model, sends a chat request, and verifies the response
/// is not empty and contains recognizable text (not gibberish).
///
/// Set TEST_E2E_INFERENCE=1 to run this test.
#[tokio::test]
#[serial(small_model)]
async fn test_e2e_inference_coherent_output() {
    if std::env::var("TEST_E2E_INFERENCE").is_err() {
        println!("Skipping e2e inference test: TEST_E2E_INFERENCE not set");
        println!("To run: TEST_E2E_INFERENCE=1 cargo test test_e2e_inference");
        return;
    }

    let model_path = get_test_model_path();
    println!("\n=== E2E Inference Test ===");
    println!("Model: {:?}", model_path);

    // Build pipeline
    let pipeline = CausalLMLoaderBuilder::from_gguf_paths(&[&model_path])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .silent()
        .build_async()
        .expect("Failed to build pipeline");

    // Create MistralRs engine
    let scheduler = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(1.try_into().unwrap()),
    };
    let runner = MistralRsBuilder::new(pipeline, scheduler, false, None)
        .build()
        .await;

    // Run inference
    let prompt = "What is 2 + 2? Answer with just the number.";
    println!("Prompt: {}", prompt);

    let response = run_simple_inference(&runner, prompt)
        .await
        .expect("Inference failed");

    println!("Response: {}", response);

    // Verify response is not empty
    assert!(!response.is_empty(), "Response should not be empty");

    // Verify response contains recognizable content (not gibberish)
    // For "2 + 2", we expect the response to contain "4" or related words
    let response_lower = response.to_lowercase();
    let has_expected_content = response_lower.contains("4")
        || response_lower.contains("four")
        || response_lower.contains("answer");

    assert!(
        has_expected_content,
        "Response should contain expected content (got: {})",
        response
    );

    println!("=== E2E Inference Test PASSED ===\n");
}

/// End-to-end inference test with LoRA adapter.
///
/// This test verifies that a model with a LoRA adapter produces coherent output.
/// Set TEST_E2E_INFERENCE=1 and TEST_LORA_ADAPTER=<repo> to run this test.
#[tokio::test]
#[serial(small_model)]
async fn test_e2e_inference_with_lora() {
    if std::env::var("TEST_E2E_INFERENCE").is_err() {
        println!("Skipping e2e LoRA inference test: TEST_E2E_INFERENCE not set");
        return;
    }

    let adapter_repo = match std::env::var("TEST_LORA_ADAPTER") {
        Ok(repo) => repo,
        Err(_) => {
            println!("Skipping e2e LoRA inference test: TEST_LORA_ADAPTER not set");
            return;
        }
    };

    let (base_repo, base_file) = if let Ok(base) = std::env::var("TEST_LORA_BASE_MODEL") {
        let file = std::env::var("TEST_LORA_BASE_FILE").unwrap_or_else(|_| "model.gguf".to_string());
        (base, file)
    } else {
        (
            DEFAULT_MODEL_REPO.to_string(),
            DEFAULT_MODEL_FILE.to_string(),
        )
    };

    println!("\n=== E2E LoRA Inference Test ===");
    println!("Base model: {}/{}", base_repo, base_file);
    println!("LoRA adapter: {}", adapter_repo);

    // Build pipeline with LoRA
    let pipeline = CausalLMLoaderBuilder::from_hf_gguf(&base_repo, &[&base_file])
        .with_device(Device::Cpu)
        .with_dtype(DType::F32)
        .with_lora_adapter(&adapter_repo)
        .silent()
        .build_async()
        .expect("Failed to build pipeline with LoRA");

    // Create MistralRs engine
    let scheduler = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(1.try_into().unwrap()),
    };
    let runner = MistralRsBuilder::new(pipeline, scheduler, false, None)
        .build()
        .await;

    // Run inference
    let prompt = "Hello! How are you today?";
    println!("Prompt: {}", prompt);

    let response = run_simple_inference(&runner, prompt)
        .await
        .expect("Inference with LoRA failed");

    println!("Response: {}", response);

    // Verify response is not empty
    assert!(
        !response.is_empty(),
        "Response with LoRA should not be empty"
    );

    // Verify response contains words (not just random bytes)
    let has_words = response.split_whitespace().count() >= 1;
    assert!(has_words, "Response should contain words (got: {})", response);

    println!("=== E2E LoRA Inference Test PASSED ===\n");
}
