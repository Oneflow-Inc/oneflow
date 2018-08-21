#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {
template<typename T>
class ProposalTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProposalTargetKernel);
  ProposalTargetKernel() = default;
  ~ProposalTargetKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void RoisNearestGtAndMaxIou(const int64_t rois_num, const T* rpn_rois_ptr,
                              const FloatList16* gt_boxes_ptr, int32_t* roi_nearest_gt_index_ptr,
                              T* roi_max_overlap_ptr) const;
  ScoredBBoxSlice<T> ForegroundChoice(ScoredBBoxSlice<T>& rois_slice) const;
  ScoredBBoxSlice<T> BackgroundChoice(ScoredBBoxSlice<T>& rois_slice,
                                      const int64_t fg_sample_size) const;
  void CopyRoIs(const ScoredBBoxSlice<T>& slice, T* rois_ptr) const;

  void ComputeTargetAndWriteOut(const ScoredBBoxSlice<T>& fg_slice,
                                const ScoredBBoxSlice<T>& bg_slice,
                                const int32_t* roi_nearest_gt_index_ptr,
                                const FloatList16* gt_boxes_ptr, const Int32List16* gt_labels_ptr,
                                T* rois_ptr, int32_t* labels_ptr, T* bbox_targets_ptr,
                                T* inside_weights_ptr, T* outside_weights_ptr) const;
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
