#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ReluForwardGpu(const int64_t n, const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : 0; }
}

template<typename T>
__global__ void ReluBackwardGpu(const int64_t n, const T* out_diff, const T* in,
                                T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_diff[i] = in[i] > 0 ? out_diff[i] : 0; }
}

}  // namespace

template<typename T>
class ReluKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n, const T* in,
                      T* out) {
    ReluForwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(n, in, out);
  }

  static void Backward(const KernelCtx& ctx, const int64_t n, const T* out_diff,
                       const T* in, T* in_diff) {
    ReluBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, out_diff, in, in_diff);
  }
};

#define DECLARE_RELU_KERNEL_UTIL(type_cpp, type_proto) \
  template class ReluKernelUtil<DeviceType::kGPU, type_cpp>;
FOR_EACH_PAIR(DECLARE_RELU_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_PAIR())

}  // namespace oneflow
