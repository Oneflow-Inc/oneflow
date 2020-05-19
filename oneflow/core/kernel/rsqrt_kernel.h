#ifndef ONEFLOW_CORE_KERNEL_RSQRT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RSQRT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class RsqrtKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RsqrtKernel);
  RsqrtKernel() = default;
  ~RsqrtKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RSQRT_KERNEL_H_
