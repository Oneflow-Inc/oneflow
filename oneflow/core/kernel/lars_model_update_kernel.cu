#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GetLocalLearningRateGpu(const float* learning_rate, T weight_decay, T epsilon,
                                        T lars_coefficient, const int64_t* train_step,
                                        T* data_tmp) {
  T* model_norm = &data_tmp[0];
  T* model_diff_norm = &data_tmp[1];
  T* local_learning_rate = &data_tmp[2];
  *model_norm = std::sqrt(*model_norm);
  *model_diff_norm = std::sqrt(*model_diff_norm);  // TODO(shiyuan)
  if (*train_step == 0) {
    *local_learning_rate =
        *learning_rate * lars_coefficient * (*model_norm) / (epsilon + (*model_diff_norm));
  } else {
    *local_learning_rate = *learning_rate * lars_coefficient * (*model_norm)
                           / (epsilon + (*model_diff_norm) + weight_decay * (*model_diff_norm));
  }
}

template<typename T>
__global__ void UpdateModelGpu(int64_t n, T weight_decay, T momentum_beta, const T* model_diff,
                               T* model, T* momentum, T* data_tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T reg_diff = model_diff[i] + weight_decay * model[i];
    momentum[i] = momentum_beta * momentum[i] - data_tmp[2] * reg_diff;
    model[i] = model[i] + momentum[i];
  }
}

}  // namespace

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T momentum_beta, T epsilon, T lars_coefficient, const int64_t* train_step,
                          const T* model_diff, T* model, T* momentum, T* data_tmp) {
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model, 1, model, 1, &data_tmp[0]);
    KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, &data_tmp[1]);
    GetLocalLearningRateGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(
        learning_rate, weight_decay, epsilon, lars_coefficient, train_step, data_tmp);
    UpdateModelGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, weight_decay, momentum_beta, model_diff, model, momentum, data_tmp);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LARSMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
