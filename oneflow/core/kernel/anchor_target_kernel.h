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
  using BoxesWithMaxOverlap = MaxOverlapIndex<BoxesIndex<T>>;
  using BoxesLabelAndMaxOverlap = LabelIndex<BoxesWithMaxOverlap>;

  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetKernel);
  AnchorTargetKernel() = default;
  ~AnchorTargetKernel() = default;

 private:
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  BoxesLabelAndMaxOverlap GetImageAnchorBoxes(
      const KernelCtx& ctx, size_t im_index,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  GtBoxesWithMaxOverlap GetImageGtBoxes(
      size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ComputeOverlapsAndSetLabels(GtBoxesWithMaxOverlap& gt_boxes,
                                   BoxesLabelAndMaxOverlap& anchor_boxes) const;
  size_t SubsampleForeground(BoxesLabelAndMaxOverlap& boxes) const;
  size_t SubsampleBackground(size_t fg_cnt, BoxesLabelAndMaxOverlap& boxes) const;
  size_t ChoiceForeground(BoxesLabelAndMaxOverlap& boxes) const;
  size_t ChoiceBackground(size_t fg_cnt, BoxesLabelAndMaxOverlap& boxes) const;
  void ComputeTargetsAndWriteOutput(
      size_t im_index, size_t total_sample_count, const GtBoxesWithMaxOverlap& gt_boxes,
      const BoxesLabelAndMaxOverlap& anchor_boxes,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
