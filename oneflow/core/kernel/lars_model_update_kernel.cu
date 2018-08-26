#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SumOfSquareGpu(int64_t n, const T* x, T* result) {
  CUDA_1D_KERNEL_LOOP(i, n) { *result += x[i] * x[i]; }
}

template<typename T>
__global__ void GetLocalLearningRateGpu(int64_t n, int64_t batch_size, T learning_rate, T l2,
                                        T epsilon, T lars_coefficient, int64_t next_model_vid,
                                        T* data_tmp) {
  T* model_norm = &data_tmp[0];
  T* model_diff_norm = &data_tmp[1];
  T* local_learning_rate = &data_tmp[2];
  *model_norm = std::sqrt(*model_norm / n);
  *model_diff_norm = std::sqrt(*model_diff_norm / n);
  if (next_model_vid == 1) {
    *local_learning_rate =
        learning_rate * lars_coefficient * (*model_norm) / (epsilon + (*model_diff_norm));
  } else {
    *local_learning_rate = learning_rate * lars_coefficient * (*model_norm)
                           / (epsilon + (*model_diff_norm) + l2 * (*model_norm));
  }
}

}  // namespace

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, int64_t batch_size, T learning_rate, T l1,
                          T l2, T momentum_beta, T epsilon, T lars_coefficient,
                          int64_t next_model_vid, const T* pre_model, const T* model_diff,
                          T* momentum, T* model, T* data_tmp) {
    SumOfSquareGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, pre_model, &data_tmp[0]);
    SumOfSquareGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, model_diff, &data_tmp[1]);
    GetLocalLearningRateGpu<T>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, batch_size, learning_rate, l2, epsilon, lars_coefficient, next_model_vid, data_tmp);
    CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
    T local_learning_rate;
    CudaCheck(cudaMemcpy(&local_learning_rate, &data_tmp[2], sizeof(T), cudaMemcpyDeviceToHost));
    NormalMdUpdateKernelUtil<DeviceType::kGPU, T>::UpdateModel(
        ctx, n, batch_size, local_learning_rate, l1, l2, momentum_beta, pre_model, model_diff,
        momentum, model);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class LARSMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
