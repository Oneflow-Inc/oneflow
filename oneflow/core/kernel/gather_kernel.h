#ifndef ONEFLOW_CORE_KERNEL_GATHER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_GATHER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class GatherKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherKernel);
  GatherKernel() = default;
  ~GatherKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct LookupKernelUtil final {
  static void Forward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices, const T* in,
                      int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_rows, int64_t in_cols, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GATHER_KERNEL_H_
