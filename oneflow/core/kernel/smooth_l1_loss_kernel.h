#ifndef ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SmoothL1LossKernel final : public LossKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1LossKernel);
  SmoothL1LossKernel() = default;
  ~SmoothL1LossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename T>
struct SmoothL1LossKernelUtil {
  static void Forward(DeviceCtx* ctx, const int32_t elem_cnt, const T* prediction, const T* label,
                      const T* inside_weights, const T* outside_weights, const float beta,
                      const float scale, T* loss);
  static void Backward(DeviceCtx* ctx, const int32_t elem_cnt, const T* predict, const T* label,
                       const T* inside_weights, const T* outside_weights, const float beta,
                       const float scale, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SMOOTH_L1_LOSS_KERNEL_H_
