#ifndef ONEFLOW_CORE_KERNEL_FPN_DISTRIBUTE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FPN_DISTRIBUTE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class FpnDistributeKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FpnDistributeKernel);
  FpnDistributeKernel() = default;
  ~FpnDistributeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdxInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FPN_DISTRIBUTE_KERNEL_H_
