#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class BboxNmsKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxNmsKernel);
  BboxNmsKernel() = default;
  ~BboxNmsKernel() = default;

  using BBox = BBoxT<const T>;
  using BBoxSlice = BBoxIndices<IndexSequence, BBox>;
  template<size_t N>
  using NDIMS = std::integral_constant<size_t, N>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  std::vector<std::vector<int32_t>> ApplyNms(
      const Blob* score_blob, const Blob* bbox_blob,
      const std::vector<std::vector<int32_t>>& class2score_id_vec) const;
  std::vector<std::vector<int32_t>> ConcatAllClasses(
      const std::vector<std::vector<int32_t>>& im2class_id_vec,
      const std::vector<std::vector<int32_t>>& class2bbox_id_vec) const;
  void GroupByImRecordAndClass(NDIMS<2>, const Blob* score_blob,
                               std::vector<std::vector<int32_t>>& im2class_id_vec,
                               std::vector<std::vector<int32_t>>& class2score_id_vec) const;
  void GroupByImRecordAndClass(NDIMS<3>, const Blob* score_blob,
                               std::vector<std::vector<int32_t>>& im2class_id_vec,
                               std::vector<std::vector<int32_t>>& class2score_id_vec) const;
  void WriteToOutput(NDIMS<2>, const std::vector<std::vector<int32_t>>& im2score_id_vec,
                     const Blob* bbox_blob, const Blob* score_blob, Blob* out_bbox_blob,
                     Blob* out_score_blob, Blob* out_label_blob) const;
  void WriteToOutput(NDIMS<3>, const std::vector<std::vector<int32_t>>& im2score_id_vec,
                     const Blob* bbox_blob, const Blob* score_blob, Blob* out_bbox_blob,
                     Blob* out_score_blob, Blob* out_label_blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_KERNEL_H_