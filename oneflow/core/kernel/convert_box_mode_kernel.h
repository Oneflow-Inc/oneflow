#ifndef ONEFLOW_CORE_OPERATOR_CONVERT_BOX_MODE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONVERT_BOX_MODE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvertBoxModeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvertBoxModeKernel);
  ConvertBoxModeKernel() = default;
  ~ConvertBoxModeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct ConvertBoxModeUtil {
  static void Forward();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONVERT_BOX_MODE_KERNEL_OP_H_
