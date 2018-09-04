#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {
template<typename T>
class ProposalTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  using BoxesWithMaxOverlap = MaxOverlapIndex<BoxesIndex<T>>;

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
  GtBoxesAndLabels GetImageGtBoxes(
      size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesWithMaxOverlap GetImageRoiBoxes(
      size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ComputeRoiBoxesAndGtBoxesOverlaps(const GtBoxesAndLabels& gt_boxes,
                                         BoxesWithMaxOverlap& roi_boxes) const;
  void ConcatGtBoxesToRoiBoxesHead(const GtBoxesAndLabels& gt_boxes,
                                   BoxesWithMaxOverlap& roi_boxes) const;
  void ConcatGtBoxesToRoiBoxesTail(const GtBoxesAndLabels& gt_boxes,
                                   BoxesWithMaxOverlap& roi_boxes) const;
  void SubsampleForegroundAndBackground(const GtBoxesAndLabels& gt_boxes,
                                        BoxesWithMaxOverlap& boxes) const;
  void ComputeAndWriteOutput(size_t im_index, const GtBoxesAndLabels& gt_boxes,
                             const BoxesWithMaxOverlap& boxes,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
