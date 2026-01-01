//! GPU profiling support for NVIDIA Nsight Systems.
//!
//! This module provides NVTX (NVIDIA Tools Extension) markers for profiling
//! CUDA operations. When the `profiling-cuda` feature is enabled, these markers
//! appear in Nsight Systems traces, allowing fine-grained analysis of inference
//! performance.
//!
//! When the feature is disabled, all operations are no-ops with zero overhead.
//!
//! ## Usage
//!
//! ```ignore
//! use mistralrs_core::profiling;
//!
//! // Mark a range of execution
//! let _range = profiling::range("model_forward");
//!
//! // Or use the macro for scoped ranges
//! profiling::nvtx_range!("attention_layer");
//! ```

#[cfg(feature = "profiling-cuda")]
pub use nvtx::{mark, range};

#[cfg(feature = "profiling-cuda")]
pub use nvtx::Range;

/// Create an NVTX range that automatically ends when dropped.
///
/// When `profiling-cuda` is disabled, this is a no-op.
#[cfg(feature = "profiling-cuda")]
#[macro_export]
macro_rules! nvtx_range {
    ($name:expr) => {
        let _nvtx_range = $crate::profiling::range($name);
    };
}

/// Create an NVTX range that automatically ends when dropped.
///
/// When `profiling-cuda` is disabled, this is a no-op.
#[cfg(not(feature = "profiling-cuda"))]
#[macro_export]
macro_rules! nvtx_range {
    ($name:expr) => {};
}

/// Place an NVTX marker at a specific point in the trace.
///
/// When `profiling-cuda` is disabled, this is a no-op.
#[cfg(feature = "profiling-cuda")]
#[macro_export]
macro_rules! nvtx_mark {
    ($name:expr) => {
        $crate::profiling::mark($name);
    };
}

/// Place an NVTX marker at a specific point in the trace.
///
/// When `profiling-cuda` is disabled, this is a no-op.
#[cfg(not(feature = "profiling-cuda"))]
#[macro_export]
macro_rules! nvtx_mark {
    ($name:expr) => {};
}

/// A no-op range guard when profiling is disabled.
#[cfg(not(feature = "profiling-cuda"))]
pub struct Range;

#[cfg(not(feature = "profiling-cuda"))]
impl Range {
    #[inline(always)]
    pub fn new(_name: &str) -> Self {
        Self
    }
}

/// Create an NVTX range (no-op when `profiling-cuda` is disabled).
#[cfg(not(feature = "profiling-cuda"))]
#[inline(always)]
pub fn range(_name: &str) -> Range {
    Range
}

/// Place an NVTX marker (no-op when `profiling-cuda` is disabled).
#[cfg(not(feature = "profiling-cuda"))]
#[inline(always)]
pub fn mark(_name: &str) {}

pub use nvtx_mark;
pub use nvtx_range;
