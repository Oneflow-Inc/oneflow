#ifndef ONEFLOW_CORE_KERNEL_SMOOTH_L1_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SMOOTH_L1_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SmoothL1Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1Kernel);
  SmoothL1Kernel() = default;
  ~SmoothL1Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct SmoothL1KernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* prediction, const T* label,
                      const float beta, const float scale, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* out_diff,
                       const T* prediction, const T* label, const float beta, const float scale,
                       T* prediction_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SMOOTH_L1_KERNEL_H_
