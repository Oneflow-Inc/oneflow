#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastSubKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSubKernel);
  BroadcastSubKernel() = default;
  ~BroadcastSubKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
