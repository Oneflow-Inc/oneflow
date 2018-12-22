#ifndef ONEFLOW_CORE_KERNEL_SSD_MULTIBOX_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SSD_MULTIBOX_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class SSDMultiboxTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SSDMultiboxTargetKernel);
  SSDMultiboxTargetKernel() = default;
  ~SSDMultiboxTargetKernel() = default;

  using BBox = BBoxImpl<const T, BBoxCategory::kFloatingLTRB>
  using BoxesSlice = BBoxIndices<IndexSequence, BBox>;
  using BoxesWithMaxOverlapSlice = MaxOverlapIndices<BoxesSlice>;
  using LabeledBoxesWithMaxOverlapSlice = LabelIndices<BoxesWithMaxOverlapSlice>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  // void InitOutputBlob(const KernelCtx& ctx,
  //                     std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  // GtBoxesAndLabels GetImageGtBoxes(
  //     size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // BoxesWithMaxOverlap GetImageRoiBoxes(
  //     size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // void ComputeRoiBoxesAndGtBoxesOverlaps(const GtBoxesAndLabels& gt_boxes,
  //                                        BoxesWithMaxOverlap& roi_boxes, float* overlaps_ptr) const;
  // int32_t SelectPosNegSample(const KernelCtx& ctx, size_t im_index, BoxesWithMaxOverlap& roi_boxes,
  //                            const GtBoxesAndLabels& gt_boxes,
  //                            const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // void SelectPositive(const BoxesWithMaxOverlap& roi_boxes, Indexes& pos_sample) const;
  // void SelectNegtive(const KernelCtx& ctx, size_t im_index, const int32_t num_pos,
  //                    const BoxesWithMaxOverlap& roi_boxes, const GtBoxesAndLabels& gt_boxes,
  //                    Indexes& neg_sample,
  //                    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // void ComputeSoftmaxLoss(const KernelCtx& ctx, size_t im_index,
  //                         const BoxesWithMaxOverlap& roi_boxes, const GtBoxesAndLabels& gt_boxes,
  //                         const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // void Output(int32_t im_index, const BoxesWithMaxOverlap& roi_boxes,
  //             const GtBoxesAndLabels& gt_boxes, const int32_t pos_num,
  //             std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  // void CopyBboxDelta(const BBoxDelta<T>* input, BBoxDelta<T>* output) const;
  // void CopyElements(const int32_t num, const T* input, T* output) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SSD_MULTIBOX_TARGET_KERNEL_H_
