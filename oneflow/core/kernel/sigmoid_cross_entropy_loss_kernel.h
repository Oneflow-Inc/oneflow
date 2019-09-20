#ifndef ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SigmoidCrossEntropyLossKernel final : public LossKernel<device_type, PredType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyLossKernel);
  SigmoidCrossEntropyLossKernel() = default;
  ~SigmoidCrossEntropyLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossKernelUtil {
  static void Forward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                      const PredType* prediction, const LabelType* label,
                      PredType* elementwise_loss, PredType* count);
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label, PredType* pred_diff);
  static void ClipByEpsilon(DeviceCtx* ctx, const int64_t n, PredType* x);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_H_
