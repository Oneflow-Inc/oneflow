#ifndef ONEFLOW_CORE_KERNEL_FPN_COLLECT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FPN_COLLECT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class FpnCollectKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FpnCollectKernel);
  FpnCollectKernel() = default;
  ~FpnCollectKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  int64_t ConcatAllRoisAndScores(const KernelCtx& ctx, const int32_t,
                              const std::function<Blob*(const std::string&)>&) const;
  void SortAndSelectTopnRois(const size_t, const std::function<Blob*(const std::string&)>&) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_