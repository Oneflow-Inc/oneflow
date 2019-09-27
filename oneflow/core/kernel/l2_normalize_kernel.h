#ifndef ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L2NormalizeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeKernel);
  L2NormalizeKernel() = default;
  ~L2NormalizeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct L2NormalizeKernelUtil {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* square_x_sum_blob, Blob* out_blob);
  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                       const Blob* out_diff_blob, const Blob* square_x_sum_blob,
                       Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_
