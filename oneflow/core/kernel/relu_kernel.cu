#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void ReluForwardGpu(const int64_t n, const FloatingPointType* in,
                               FloatingPointType* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : 0; }
}

template<typename FloatingPointType>
__global__ void ReluBackwardGpu(const int64_t n,
                                const FloatingPointType* out_diff,
                                const FloatingPointType* in,
                                FloatingPointType* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_diff[i] = in[i] > 0 ? out_diff[i] : 0; }
}

}  // namespace

template<typename FloatingPointType>
class ReluKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n,
                      const FloatingPointType* in, FloatingPointType* out) {
    ReluForwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, in, out);
    CudaPostKernelCheck();
  }

  static void Backward(const KernelCtx& ctx, const int64_t n,
                       const FloatingPointType* out_diff,
                       const FloatingPointType* in,
                       FloatingPointType* in_diff) {
    ReluBackwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, out_diff, in, in_diff);
    CudaPostKernelCheck();
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(ReluKernelUtil);

}  // namespace oneflow
