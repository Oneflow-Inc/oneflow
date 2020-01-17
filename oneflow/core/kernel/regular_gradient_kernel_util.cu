#include "oneflow/core/kernel/regular_gradient_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void RegularGradientGpu(int64_t n, const T* model, const T* model_diff, T* out,
                                   const T l1_scale, const T l2_scale) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_val = model[i];
    out[i] =
        model_diff[i] + l1_scale * ((model_val >= 0) - (model_val <= 0)) + l2_scale * model_val;
  }
}

}  // namespace

template<typename T>
struct RegularGradientKernelUtil<DeviceType::kGPU, T> {
  static void RegularGradient(DeviceCtx* ctx, int64_t n, const T* model, const T* model_diff,
                              T* out, const T l1_scale, const T l2_scale) {
    RegularGradientGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, model, model_diff, out, l1_scale, l2_scale);
  }
};

#define INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct RegularGradientKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_GPU

}  // namespace oneflow
