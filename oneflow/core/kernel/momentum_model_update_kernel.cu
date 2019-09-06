#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpdateModelGpu(int64_t n, const T* batch_instance_num_ptr, T beta,
                               const int64_t* train_step, const float* learning_rate, T l1, T l2,
                               const T* model_diff, T* model, T* momentum) {
  T cur_beta = *train_step == 0 ? 0 : beta;
  CUDA_1D_KERNEL_LOOP(i, n) {
    T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
    momentum[i] = cur_beta * momentum[i] - *learning_rate * reg_diff;
    model[i] = model[i] + momentum[i];
  }
}

}  // namespace

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr, T beta,
                          const int64_t* train_step, const float* learning_rate, const T l1,
                          const T l2, const T* model_diff, T* model, T* momentum) {
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, batch_instance_num_ptr, beta, train_step, learning_rate, l1, l2, model_diff, model,
        momentum);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class MomentumMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
