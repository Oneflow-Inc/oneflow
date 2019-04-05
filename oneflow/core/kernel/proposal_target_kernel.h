#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class ProposalTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProposalTargetKernel);
  ProposalTargetKernel() = default;
  ~ProposalTargetKernel() = default;

  using BBox = IndexedBBoxT<const T>;
  using MutBBox = IndexedBBoxT<T>;
  using GtBBox = BBoxT<const T>;
  using MutGtBBox = BBoxT<T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void InitializeOutputBlob(DeviceCtx* ctx,
                            const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void GenMatchMatrixBetweenRoiAndGtBoxes(
      DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void Subsample(DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void Output(DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
