#ifndef ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename CondType, typename T>
class WhereKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereKernel);
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename CondType, typename T>
struct WhereKernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_prt,
                      const T* lhs_ptr, const T* rhs_ptr, T* out_ptr);
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_prt,
                       const T* out_diff_ptr, T* lhs_diff_ptr, T* rhs_diff_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_
