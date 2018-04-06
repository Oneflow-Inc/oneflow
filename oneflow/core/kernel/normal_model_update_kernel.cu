#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void DiffAveragingL1RegularizationGpu(int64_t n, T l1,
                                                 int32_t batch_size, const T* x,
                                                 T* y) {
  T zero = ZeroVal<T>::value;
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = y[i] / batch_size + l1 * ((x[i] >= zero) - (x[i] <= zero));
  }
}

}  // namespace
template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void DiffAveragingAndL1Regularization(DeviceCtx* ctx, int64_t n,
                                               float l1, const T* model,
                                               T* model_diff_acc) {
    int32_t batch_size = Global<JobDesc>::Get()->BatchSize();
    DiffAveragingL1RegularizationGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, static_cast<T>(l1), batch_size, model,
                                 model_diff_acc);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NormalMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
