#include "oneflow/core/kernel/clip_by_value_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t elem_cnt, const T* in_ptr, const T min_val,
                           const T max_val, int8_t* clip_mask_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_ptr[i] = min(max(in_ptr[i], min_val), max_val);
    clip_mask_ptr[i] = (out_ptr[i] == in_ptr[i]);
  }
}

template<typename T>
__global__ void GpuBackward(const int64_t elem_cnt, const T* out_diff_ptr,
                            const int8_t* clip_mask_ptr, T* in_diff_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { in_diff_ptr[i] = out_diff_ptr[i] * clip_mask_ptr[i]; }
}

}  // namespace

template<typename T>
struct ClipByValueUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* in_ptr, const T min_val,
                      const T max_val, int8_t* clip_mask_ptr, T* out_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, in_ptr, min_val, max_val, clip_mask_ptr, out_ptr);
  }
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* out_diff_ptr,
                       const int8_t* clip_mask_ptr, T* in_diff_ptr) {
    GpuBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, out_diff_ptr, clip_mask_ptr, in_diff_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ClipByValueUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
