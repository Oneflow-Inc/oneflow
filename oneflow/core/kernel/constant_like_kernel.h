#ifndef ONEFLOW_CORE_OPERATOR_CONSTANT_LIKE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONSTANT_LIKE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConstantLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantLikeKernel);
  ConstantLikeKernel() = default;
  ~ConstantLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct ConstantLikeUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, T* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONSTANT_LIKE_KERNEL_OP_H_
