#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {

template<typename T>
class AnchorTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  using BoxesLabelsAndNearestGtBoxes = LabelSlice<BoxesToNearestGtBoxesSlice<BoxesSlice<T>>>;

  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetKernel);
  AnchorTargetKernel() = default;
  ~AnchorTargetKernel() = default;

 private:
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  BoxesSlice<T> GetAnchorBoxesSlice(
      const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesSlice<T> GetImageGtBoxesSlice(
      size_t image_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  BoxesLabelsAndNearestGtBoxes ComputeOverlapsAndSetLabels(
      size_t image_index, const BoxesSlice<T>& gt_boxes_slice,
      const BoxesSlice<T>& anchor_boxes_slice,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  size_t SubsampleForeground(BoxesLabelsAndNearestGtBoxes& boxes_labels_and_nearest_gt_boxes) const;
  size_t SubsampleBackground(size_t fg_cnt,
                             BoxesLabelsAndNearestGtBoxes& boxes_labels_and_nearest_gt_boxes) const;
  void WriteOutput(size_t image_index, size_t total_sample_count,
                   const BoxesLabelsAndNearestGtBoxes& boxes_labels_and_nearest_gt_boxes,
                   const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
