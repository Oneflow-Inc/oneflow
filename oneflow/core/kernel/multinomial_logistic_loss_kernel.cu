#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void MultinomialLogisticLossForwardGpu(const int64_t instance_num,
                                                  const int64_t num_of_classes,
                                                  const PredType* prediction,
                                                  const LabelType* labels,
                                                  PredType* loss_buff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    PredType prob =
        prediction[i * num_of_classes + static_cast<int64_t>(labels[i])];
    loss_buff[i] = -SAFE_LOG(prob);
  }
}

template<typename PredType, typename LabelType>
__global__ void MultinomialLogisticLossBackwardGpu(const int64_t instance_num,
                                                   const int64_t num_of_classes,
                                                   const PredType* prediction,
                                                   const LabelType* labels,
                                                   PredType* prediction_diff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    PredType prob = prediction[i * num_of_classes + label];
    prediction_diff[i * num_of_classes + label] =
        -1 / MAX_WITH_LOG_THRESHOLD(prob);
  }
}

}  // namespace

template<typename PredType, typename LabelType>
class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, PredType, LabelType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const PredType* prediction,
                      const LabelType* labels, PredType* loss,
                      PredType* loss_buff) {
    MultinomialLogisticLossForwardGpu<PredType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, num_of_classes, prediction,
                                 labels, loss_buff);
    KernelUtil<DeviceType::kGPU, PredType>::Sum(
        ctx, instance_num, loss_buff, loss, loss_buff,
        sizeof(PredType) * instance_num);
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const PredType* prediction,
                       const LabelType* labels, PredType* prediction_diff) {
    MultinomialLogisticLossBackwardGpu<PredType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, num_of_classes, prediction,
                                 labels, prediction_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)       \
  template class MultinomialLogisticLossKernelUtil<       \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
      OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
