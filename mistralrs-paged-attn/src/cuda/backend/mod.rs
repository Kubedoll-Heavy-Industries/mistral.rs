mod cache;
mod paged_attention;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, DevicePtr, DeviceRepr},
};
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;

pub fn fp8_supported_on_device(device: &candle_core::cuda::CudaDevice) -> bool {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    use candle_core::cuda_backend::WrapErr;

    let Ok(()) = device.cuda_stream().context().bind_to_thread().w() else {
        return false;
    };

    let Ok(major) = device
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .w()
    else {
        return false;
    };

    major >= 8
}

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}
