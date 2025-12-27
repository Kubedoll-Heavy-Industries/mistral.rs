pub fn fp8_supported_on_device(device: &candle_core::cuda::CudaDevice) -> bool {
    backend::fp8_supported_on_device(device)
}

mod backend;
mod ffi;

pub use backend::{copy_blocks, kv_scale_update, paged_attention, reshape_and_cache, swap_blocks};
