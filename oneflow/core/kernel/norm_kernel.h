#ifndef ONEFLOW_CORE_KERNEL_NORM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormKernel);
  NormKernel() = default;
  ~NormKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct NormKernelUtil {
  static void Abs(DeviceCtx* ctx, const int32_t n, const T epsilon, const T* in_dptr,
                  T* abs_tmp_dptr);
  static void L1NormBackward(DeviceCtx* ctx, const int32_t out_n, const int32_t offset,
                             const T* out_diff_dptr, const T* in_dptr, T* in_diff_dptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORM_KERNEL_H_
