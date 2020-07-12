#include "oneflow/core/kernel/l1_l2_regularize_gradient_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void L1L2RegularizeGradientGpu(int64_t n, const T* model, const T* model_diff, T* out,
                                          const T l1, const T l2) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_val = model[i];
    out[i] = model_diff[i] + l1 * ((model_val >= 0) - (model_val <= 0)) + l2 * model_val;
  }
}

}  // namespace

template<typename T>
struct L1L2RegularizeGradientKernelUtil<DeviceType::kGPU, T> {
  static void RegularizeGradient(DeviceCtx* ctx, int64_t n, const T* model, const T* model_diff,
                                 T* out, const T l1, const T l2) {
    L1L2RegularizeGradientGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(n, model, model_diff, out, l1, l2);
  }
};

#define INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct L1L2RegularizeGradientKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_GPU

}  // namespace oneflow
