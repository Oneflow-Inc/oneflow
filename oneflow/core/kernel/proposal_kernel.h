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

  using BBox = BBoxImpl<const T, BBoxCategory::kCorner>;
  using MutBBox = BBoxImpl<T, BBoxCategory::kCorner>;
  using RoiBox = BBoxImpl<T, BBoxCategory::kIndexCorner>;
  using BoxesSlice = BBoxIndices<IndexSequence, BBox>;
  using ScoreSlice = ScoreIndices<IndexSequence, const T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdxInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  ScoreSlice RegionProposal(const int64_t im_index,
                            const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesSlice ApplyNms(const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  size_t WriteRoisToOutput(const size_t num_output, const int32_t im_index,
                           const ScoreSlice& score_slice, const BoxesSlice& post_nms_slice,
                           const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
