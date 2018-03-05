#ifndef ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LocalResponseNormalizationKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalResponseNormalizationKernel);
  LocalResponseNormalizationKernel() = default;
  ~LocalResponseNormalizationKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_
