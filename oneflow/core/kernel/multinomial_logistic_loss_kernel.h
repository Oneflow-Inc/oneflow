#ifndef ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
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

template<DeviceType device_type, typename FloatingPointType>
class MultinomialLogisticLossKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t instance_num,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels, FloatingPointType* loss,
                      FloatingPointType* loss_buff);
  static void Backward(const KernelCtx& ctx, const int64_t instance_num,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MULTINOMIAL_LOGISTIC_LOSS_KERNEL_H_
