#ifndef ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SigmoidCrossEntropyLossGradKernel final : public LossKernel<device_type, PredType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyLossGradKernel);
  SigmoidCrossEntropyLossGradKernel() = default;
  ~SigmoidCrossEntropyLossGradKernel() override = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossGradKernelUtil {
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossGradOpConf& conf,
                       const int64_t n, const PredType* prediction, const LabelType* label,
                       PredType* pred_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_GRAD_KERNEL_H_
