#ifndef ONEFLOW_CORE_KERNEL_CONST_RANGE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONST_RANGE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConstRangeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstRangeKernel);
  ConstRangeKernel() : output_inited_(new bool(false)) {}
  ~ConstRangeKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;

  std::unique_ptr<bool> output_inited_;
};

template<DeviceType device_type, typename T>
struct ConstRangeKernelUtil final {
  static void Fill(DeviceCtx* ctx, T start, int64_t size, T stride, T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONST_RANGE_KERNEL_H_
