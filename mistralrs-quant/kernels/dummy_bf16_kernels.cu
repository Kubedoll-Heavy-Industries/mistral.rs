// Dummy bf16 kernel implementations for compute capability < 80
// These provide link-time symbols that assert at runtime if called.
// The HAVE_BF16_KERNELS constant in Rust prevents these from being called.

#include <cassert>
#include <cstdint>

// ============================================================================
// AFQ Dequantize stubs
// ============================================================================

#define DEFINE_DEQUANT_STUB(bits, gs) \
  extern "C" void afq_dequantize_##bits##bit_gs##gs##_bf16( \
      const void *, const void *, const void *, void *, int, int) { assert(false && "bf16 kernels require SM 80+"); }

#define DEFINE_DEQUANT_3BIT_STUB(gs) \
  extern "C" void afq_dequantize_3bit_gs##gs##_bf16( \
      const void *, const void *, const void *, void *, int, int) { assert(false && "bf16 kernels require SM 80+"); }

#define DEFINE_DEQUANT_6BIT_STUB(gs) \
  extern "C" void afq_dequantize_6bit_gs##gs##_bf16( \
      const void *, const void *, const void *, void *, int, int) { assert(false && "bf16 kernels require SM 80+"); }

DEFINE_DEQUANT_STUB(2, 32)
DEFINE_DEQUANT_STUB(2, 64)
DEFINE_DEQUANT_STUB(2, 128)
DEFINE_DEQUANT_3BIT_STUB(32)
DEFINE_DEQUANT_3BIT_STUB(64)
DEFINE_DEQUANT_3BIT_STUB(128)
DEFINE_DEQUANT_STUB(4, 32)
DEFINE_DEQUANT_STUB(4, 64)
DEFINE_DEQUANT_STUB(4, 128)
DEFINE_DEQUANT_6BIT_STUB(32)
DEFINE_DEQUANT_6BIT_STUB(64)
DEFINE_DEQUANT_6BIT_STUB(128)
DEFINE_DEQUANT_STUB(8, 32)
DEFINE_DEQUANT_STUB(8, 64)
DEFINE_DEQUANT_STUB(8, 128)

// ============================================================================
// AFQ Quantize stubs
// ============================================================================

#define DEFINE_QUANT_STUB(bits, gs) \
  extern "C" void afq_quantize_##bits##bit_gs##gs##_bf16( \
      const void *, uint32_t *, void *, void *, int, int) { assert(false && "bf16 kernels require SM 80+"); }

DEFINE_QUANT_STUB(2, 32)
DEFINE_QUANT_STUB(2, 64)
DEFINE_QUANT_STUB(2, 128)
DEFINE_QUANT_STUB(4, 32)
DEFINE_QUANT_STUB(4, 64)
DEFINE_QUANT_STUB(4, 128)
DEFINE_QUANT_STUB(8, 32)
DEFINE_QUANT_STUB(8, 64)
DEFINE_QUANT_STUB(8, 128)

// ============================================================================
// AFQ QMV stubs
// ============================================================================

#define DEFINE_QMV_STUB(bits, gs) \
  extern "C" void afq_qmv_##bits##bit_gs##gs##_bf16( \
      const void *, const uint32_t *, const void *, const void *, void *, int, int, int) { assert(false && "bf16 kernels require SM 80+"); }

#define DEFINE_QMV_3BIT_STUB(gs) \
  extern "C" void afq_qmv_3bit_gs##gs##_bf16( \
      const void *, const uint8_t *, const void *, const void *, void *, int, int, int) { assert(false && "bf16 kernels require SM 80+"); }

#define DEFINE_QMV_6BIT_STUB(gs) \
  extern "C" void afq_qmv_6bit_gs##gs##_bf16( \
      const void *, const uint8_t *, const void *, const void *, void *, int, int, int) { assert(false && "bf16 kernels require SM 80+"); }

DEFINE_QMV_STUB(2, 32)
DEFINE_QMV_STUB(2, 64)
DEFINE_QMV_STUB(2, 128)
DEFINE_QMV_3BIT_STUB(32)
DEFINE_QMV_3BIT_STUB(64)
DEFINE_QMV_3BIT_STUB(128)
DEFINE_QMV_STUB(4, 32)
DEFINE_QMV_STUB(4, 64)
DEFINE_QMV_STUB(4, 128)
DEFINE_QMV_6BIT_STUB(32)
DEFINE_QMV_6BIT_STUB(64)
DEFINE_QMV_6BIT_STUB(128)
DEFINE_QMV_STUB(8, 32)
DEFINE_QMV_STUB(8, 64)
DEFINE_QMV_STUB(8, 128)

// ============================================================================
// AFQ QMM stubs
// ============================================================================

#define DEFINE_QMM_STUB(bits, gs) \
  extern "C" void afq_qmm_##bits##bit_gs##gs##_bf16( \
      const void *, const uint32_t *, const void *, const void *, void *, int, int, int) { assert(false && "bf16 kernels require SM 80+"); }

DEFINE_QMM_STUB(2, 32)
DEFINE_QMM_STUB(2, 64)
DEFINE_QMM_STUB(2, 128)
DEFINE_QMM_STUB(4, 32)
DEFINE_QMM_STUB(4, 64)
DEFINE_QMM_STUB(4, 128)
DEFINE_QMM_STUB(8, 32)
DEFINE_QMM_STUB(8, 64)
DEFINE_QMM_STUB(8, 128)

// ============================================================================
// Bitsandbytes bf16 stubs
// ============================================================================

extern "C" void dequantize_blockwise_bf16_int8(
    float *, unsigned char *, float *, void *, int, int, void *) {
  assert(false && "bf16 kernels require SM 80+");
}

extern "C" void dequantize_blockwise_bf16_fp4(
    float *, unsigned char *, float *, void *, int, int, void *) {
  assert(false && "bf16 kernels require SM 80+");
}

extern "C" void dequantize_blockwise_bf16_nf4(
    float *, unsigned char *, float *, void *, int, int, void *) {
  assert(false && "bf16 kernels require SM 80+");
}
