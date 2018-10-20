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

  using BBox = BBoxImpl<T, BBoxCategory::kIndexCorner>;
  using RoiBox = BBoxImpl<const T, BBoxCategory::kIndexCorner>;
  using GtBox = BBoxImpl<const T, BBoxCategory::kGtCorner>;
  using RoiBoxIndices = BBoxIndices<IndexSequence, RoiBox>;
  using MaxOverlapOfRoiBoxWithGt = MaxOverlapIndices<RoiBoxIndices>;
  using GtBoxIndices = BBoxIndices<IndexSequence, GtBox>;
  using LabeledGtBox = LabelIndices<GtBoxIndices>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  LabeledGtBox GetImageGtBoxes(const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  MaxOverlapOfRoiBoxWithGt GetImageRoiBoxes(
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void FindNearestGtBoxForEachRoiBox(size_t max_num_gt_per_im, const LabeledGtBox& gt_boxes,
                                     MaxOverlapOfRoiBoxWithGt& roi_boxes) const;
  void ConcatGtBoxesToRoiBoxesHead(const LabeledGtBox& gt_boxes,
                                   MaxOverlapOfRoiBoxWithGt& roi_boxes) const;
  void SubsampleForegroundAndBackground(size_t num_im, const LabeledGtBox& gt_boxes,
                                        MaxOverlapOfRoiBoxWithGt& boxes) const;
  void Output(const LabeledGtBox& gt_boxes, const MaxOverlapOfRoiBoxWithGt& boxes,
              const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
