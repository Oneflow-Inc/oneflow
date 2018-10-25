#ifndef ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/record/coco.pb.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class MaskTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskTargetKernel);
  MaskTargetKernel() = default;
  ~MaskTargetKernel() override = default;

 private:
  using RoiBBox = IndexedBBoxT<T>;
  using SegmBBox = BBoxT<float>;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ParseSegmPolygonLists(const Blob* gt_segms,
                             std::vector<std::vector<PolygonList>>* segms) const;
  void ComputeSegmBBoxes(const std::vector<std::vector<PolygonList>>& segms, Blob* bboxes) const;
  void Segm2BBox(const PolygonList& segm, SegmBBox* bbox) const;
  void Segm2Mask(const PolygonList& segm, const RoiBBox& fg_roi, size_t mask_h, size_t mask_w,
                 int32_t* mask) const;
  void Polygon2BBox(const FloatList& polygon, SegmBBox* bbox) const;
  size_t GetMaxOverlapIndex(const RoiBBox& fg_roi, const SegmBBox* gt_bboxs,
                            size_t gt_bboxs_num) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MASK_TARGET_KERNEL_H_
