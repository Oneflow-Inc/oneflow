#ifndef ONEFLOW_CORE_KERNEL_CONCAT_GT_PROPOSALS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONCAT_GT_PROPOSALS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class ConcatGtProposalsKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatGtProposalsKernel);
  ConcatGtProposalsKernel() = default;
  ~ConcatGtProposalsKernel() = default;

  using BBox = IndexedBBoxT<T>;
  using GtBBox = BBoxT<const T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_GT_PROPOSALS_KERNEL_H_
