#ifndef ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
#define ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template <DeviceType device_type>
class XrtLaunchKernel : public KernelIf<device_type> {
 public:
  XrtLaunchKernel() = default;
  virtual ~XrtLaunchKernel() {}

 private:
  void ForwardDataContent(
      const KernelCtx &ctx,
      std::function<Blob *(const std::string &)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
