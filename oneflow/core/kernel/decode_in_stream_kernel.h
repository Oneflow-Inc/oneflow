#ifndef ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

// template<DeviceType device_type>
class DecodeInStreamKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeInStreamKernel);
  DecodeInStreamKernel() = default;
  ~DecodeInStreamKernel() = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_