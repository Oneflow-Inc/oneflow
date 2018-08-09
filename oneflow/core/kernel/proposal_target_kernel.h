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
  void RoisNearestGtAndMaxIou(const int64_t rois_num, const int64_t gt_num, const T* rpn_rois_ptr,
                              const T* gt_boxes_ptr, int32_t* roi_nearest_gt_index_ptr,
                              T* roi_max_overlap_ptr) const;
  ScoredBBoxSlice<T> ForegroundChoice(const ProposalTargetOpConf& conf,
                                      ScoredBBoxSlice<T>& rois_slice,
                                      int64_t& fg_sample_size) const;
  ScoredBBoxSlice<T> BackgroundChoice(const ProposalTargetOpConf& conf,
                                      ScoredBBoxSlice<T>& rois_slice,
                                      const int64_t fg_sample_size) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_TARGET_KERNEL_H_
