#ifndef ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MultinomialLogisticLossKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernel);
  MultinomialLogisticLossKernel() = default;
  ~MultinomialLogisticLossKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
};

template<DeviceType device_type, typename T>
class MultinomialLogisticLossKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const T* prediction,
                      const int32_t* labels, T* loss, T* loss_buff);
  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const T* prediction,
                       const int32_t* labels, T* prediction_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_
