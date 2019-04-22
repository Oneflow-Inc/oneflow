#ifndef ONEFLOW_CORE_KERNEL_LOCAL_GATHER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_GATHER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LocalGatherKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherKernel);
  LocalGatherKernel() = default;
  ~LocalGatherKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
struct LocalGatherKernelUtil final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out);
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_GATHER_KERNEL_H_
