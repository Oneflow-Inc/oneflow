#ifndef ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class OneHotKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneHotKernel)
  OneHotKernel() = default;
  ~OneHotKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
struct OneHotKernelUtil final {
  static void Encode(DeviceCtx* ctx, const K* indices, int64_t num_indices, int64_t depth, T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ONE_HOT_KERNEL_H_
