#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {
template<typename T>
class ProposalTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  using GtBoxesType = GtBoxes<FloatList16>;
  using GtBoxesWithLabelsType = GtBoxesWithLabels<FloatList16, Int32List16>;
  using BoxesWithMaxOverlapSlice = BoxesToNearestGtBoxesSlice<BoxesSlice<T>>;

  OF_DISALLOW_COPY_AND_MOVE(ProposalTargetKernel);
  ProposalTargetKernel() = default;
  ~ProposalTargetKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ClearOutputBlob(const KernelCtx& ctx,
                       std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  GtBoxesWithLabelsType GetImageGtBoxesWithLabels(
      size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesSlice<T> GetRoiBoxesSlice(size_t im_index,
                                 const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesWithMaxOverlapSlice ComputeRoiBoxesAndGtBoxesOverlaps(
      const BoxesSlice<T>& roi_boxes, const GtBoxesType& gt_boxes,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ConcatGtBoxesToRoiBoxes(const GtBoxesType& gt_boxes, BoxesSlice<T>& roi_boxes) const;
  void SubsampleForegroundAndBackground(BoxesWithMaxOverlapSlice& boxes_max_overlap) const;
  void ComputeAndWriteOutput(size_t im_index, const BoxesWithMaxOverlapSlice& boxes_slice,
                             const GtBoxesWithLabelsType& gt_boxes,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
