#ifndef ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceV2Kernel);
  PieceSliceV2Kernel() = default;
  ~PieceSliceV2Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInstanceShape(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_H_
