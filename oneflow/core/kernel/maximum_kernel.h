#ifndef ONEFLOW_CORE_KERNEL_MAXIMUM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAXIMUM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaximumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaximumKernel);
  MaximumKernel() = default;
  ~MaximumKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct MaximumKernelUtil {
  static void CWiseMaxWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int y_idx,
                               int32_t* mask);
  static void CWiseSetWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int x_idx,
                               const int32_t* mask);
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAXIMUM_KERNEL_H_
