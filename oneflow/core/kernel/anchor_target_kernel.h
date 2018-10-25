#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class AnchorTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetKernel);
  AnchorTargetKernel() = default;
  ~AnchorTargetKernel() = default;

  using BBox = BBoxT<const T>;
  using MutBBox = BBoxT<T>;
  using AnchorBoxes = BBoxIndices<IndexSequence, BBox>;
  using MaxOverlapOfBoxesWithGt = MaxOverlapIndices<AnchorBoxes>;
  using MaxOverlapOfLabeledBoxesWithGt = LabelIndices<MaxOverlapOfBoxesWithGt>;
  using GtBoxes = BBoxIndices<IndexSequence, BBox>;

 private:
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  MaxOverlapOfLabeledBoxesWithGt GetImageAnchorBoxes(
      const KernelCtx& ctx, size_t im_index,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  GtBoxes GetImageGtBoxes(size_t im_index,
                          const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;

  void CalcMaxOverlapAndSetPositiveLabels(const GtBoxes& gt_boxes,
                                          MaxOverlapOfLabeledBoxesWithGt& anchor_boxes) const;
  size_t SubsampleForeground(MaxOverlapOfLabeledBoxesWithGt& boxes) const;
  size_t SubsampleBackground(size_t fg_cnt, MaxOverlapOfLabeledBoxesWithGt& boxes) const;
  size_t ChoiceForeground(MaxOverlapOfLabeledBoxesWithGt& boxes) const;
  size_t ChoiceBackground(size_t fg_cnt, MaxOverlapOfLabeledBoxesWithGt& boxes) const;
  void OutputForEachImage(size_t im_index, size_t total_sample_cnt, const GtBoxes& gt_boxes,
                          const MaxOverlapOfLabeledBoxesWithGt& boxes,
                          const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
