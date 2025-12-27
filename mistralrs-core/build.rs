#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, process::Command, vec};
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let lib_files = vec![
            "src/cuda/sort.cu",
            "src/cuda/moe_gemm.cu",
            "src/cuda/moe_gemm_wmma.cu",
        ];
        for lib_file in lib_files.iter() {
            println!("cargo:rerun-if-changed={lib_file}");
        }
        let mut builder = bindgen_cuda::Builder::default()
            .kernel_paths(lib_files)
            .out_dir(build_dir.clone())
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose")
            .arg("--compiler-options")
            .arg("-fPIC");

        // Disable bf16 kernels for pre-sm80 devices.
        // bf16 WMMA operations and certain bf16 intrinsics are only available on sm_80+.
        let compute_cap = if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
            var.parse::<u32>().unwrap_or_else(|_| {
                panic!("Failed to parse CUDA_COMPUTE_CAP={var} as u32 (expected e.g. 75, 80, 90)")
            })
        } else {
            let out = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv"])
                .output()
                .unwrap_or_else(|_| {
                    panic!(
                        "`CUDA_COMPUTE_CAP` env var not specified and `nvidia-smi` was not found."
                    )
                });
            let output = String::from_utf8(out.stdout).expect("Output of nvidia-smi was not utf8.");
            let cap_str = output
                .lines()
                .skip(1)
                .find(|l| !l.trim().is_empty())
                .expect("nvidia-smi output did not contain a compute_cap row")
                .trim();
            (cap_str.parse::<f32>().unwrap_or_else(|_| {
                panic!("Failed to parse nvidia-smi compute_cap={cap_str} as f32")
            }) * 10.0) as u32
        };

        if compute_cap < 80 {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();

        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrscuda.lib")
        } else {
            build_dir.join("libmistralrscuda.a")
        };

        builder.build_lib(out_file);
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrscuda");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}
