#ifndef ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SmoothL1LossKernel final : public LossKernel<device_type, PredType, LabelType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1LossKernel);
  SmoothL1LossKernel() = default;
  ~SmoothL1LossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct SmoothL1LossKernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                      const PredType* prediction, const LabelType* label,
                      const PredType* inside_weights, const PredType* outside_weights,
                      const float beta, const float scale, PredType* loss);
  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                       const PredType* predict, const LabelType* target,
                       const PredType* inside_weights, const PredType* outside_weights,
                       const float beta, const float scale, PredType* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_
