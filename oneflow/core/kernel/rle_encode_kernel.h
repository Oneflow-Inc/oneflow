#ifndef ONEFLOW_CORE_KERNEL_RLE_ENCODE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RLE_ENCODE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class RleEncodeKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RleEncodeKernel);
  RleEncodeKernel() = default;
  ~RleEncodeKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RLE_ENCODE_KERNEL_H_
