#ifndef ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_

#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

class DecodeRandomKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomKernel);
  DecodeRandomKernel() = default;
  ~DecodeRandomKernel() = default;

 private:
  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
