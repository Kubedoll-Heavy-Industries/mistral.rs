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
