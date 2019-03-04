#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class ProposalKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProposalKernel);
  ProposalKernel() = default;
  ~ProposalKernel() = default;

  using BBox = BBoxT<const T>;
  using MutBBox = BBoxT<T>;
  using RoiBBox = IndexedBBoxT<T>;
  using BoxesSlice = BBoxIndices<IndexSequence, BBox>;
  using ScoreSlice = ScoreIndices<IndexSequence, const T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void GenerateAnchors(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void RegionProposal(const int64_t im_index,
                      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ApplyNms(const int64_t im_index,
                const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void WriteRoisToOutput(const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
