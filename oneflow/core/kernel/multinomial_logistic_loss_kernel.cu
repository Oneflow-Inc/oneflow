#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MultinomialLogisticLossForwardGpu(const int64_t instance_num,
                                                  const int64_t num_of_classes,
                                                  const T* prediction,
                                                  const int32_t* labels,
                                                  T* loss_buff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    T prob = prediction[i * num_of_classes + static_cast<int64_t>(labels[i])];
    loss_buff[i] = -SAFE_LOG(prob);
  }
}

template<typename T>
__global__ void MultinomialLogisticLossBackwardGpu(const int64_t instance_num,
                                                   const int64_t num_of_classes,
                                                   const T* prediction,
                                                   const int32_t* labels,
                                                   T* prediction_diff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    T prob = prediction[i * num_of_classes + label];
    prediction_diff[i * num_of_classes + label] =
        -1 / MAX_WITH_LOG_THRESHOLD(prob);
  }
}

}  // namespace

template<typename T>
class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const T* prediction,
                      const int32_t* labels, T* loss, T* loss_buff) {
    MultinomialLogisticLossForwardGpu<T>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, num_of_classes, prediction,
                                 labels, loss_buff);
    KernelUtil<DeviceType::kGPU, T>::Sum(ctx, instance_num, loss_buff, loss,
                                         loss_buff, sizeof(T) * instance_num);
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const T* prediction,
                       const int32_t* labels, T* prediction_diff) {
    MultinomialLogisticLossBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, num_of_classes, prediction,
                                 labels, prediction_diff);
  }
};

#define INSTANTIATE_M11L_LOGISTIC_LOSS_KERNEL(type_cpp, type_proto) \
  template class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, type_cpp>;
FOR_EACH_PAIR(INSTANTIATE_M11L_LOGISTIC_LOSS_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
