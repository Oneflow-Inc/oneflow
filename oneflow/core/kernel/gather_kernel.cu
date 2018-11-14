#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>
namespace oneflow {

namespace {

template<typename T>
__global__ void LookupForwardGpu(const int64_t elem_cnt, const int32_t* indices, const T* in,
                                 int64_t in_rows, int64_t in_cols, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t out_idx = i / in_cols;
    const int64_t offset = i % in_cols;
    const int64_t idx = indices[out_idx];
    assert(idx >= 0 && idx < in_rows);
    out[i] = in[idx * in_cols + offset];
  }
}

template<typename T>
__global__ void LookupBackwardGpu(const int64_t elem_cnt, const int32_t* indices, const T* out_diff,
                                  int64_t in_rows, int64_t in_cols, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t out_idx = i / in_cols;
    const int64_t offset = i % in_cols;
    const int64_t idx = indices[out_idx];
    assert(idx >= 0 && idx < in_rows);
    gpu_atomic_add(in_diff + idx * in_cols + offset, out_diff[i]);
  }
}

}  // namespace

template<typename T>
struct LookupKernelUtil<DeviceType::kGPU, T> final {
  static void Forward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices, const T* in,
                      int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_rows, int64_t in_cols, T* in_diff);
};

template<typename T>
void LookupKernelUtil<DeviceType::kGPU, T>::Forward(DeviceCtx* ctx, const int32_t* indices,
                                                    int64_t num_indices, const T* in,
                                                    int64_t in_rows, int64_t in_cols, T* out) {
  const int64_t elem_cnt = num_indices * in_cols;
  LookupForwardGpu<T>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, in, in_rows, in_cols, out);
}

template<typename T>
void LookupKernelUtil<DeviceType::kGPU, T>::Backward(DeviceCtx* ctx, const int32_t* indices,
                                                     int64_t num_indices, const T* out_diff,
                                                     int64_t in_rows, int64_t in_cols, T* in_diff) {
  const int64_t elem_cnt = num_indices * in_cols;
  LookupBackwardGpu<T>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, out_diff, in_rows, in_cols, in_diff);
}

#define INITIATE_LOOK_UP_KERNEL_UTIL(T, type_proto) \
  template struct LookupKernelUtil<DeviceType::kGPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_LOOK_UP_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);
#undef INITIATE_LOOK_UP_KERNEL_UTIL

}  // namespace oneflow
