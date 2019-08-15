#ifndef ONEFLOW_CORE_OPERATOR_LOGICAL_AND_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOGICAL_AND_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LogicalAndKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalAndKernel);
  LogicalAndKernel() = default;
  ~LogicalAndKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct LogicalAndUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* lhs_ptr, const T* rhs_ptr,
                      T* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOGICAL_AND_KERNEL_OP_H_
