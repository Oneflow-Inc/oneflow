#include "oneflow/core/register/partial_tensor_copier.h"

namespace oneflow {

PartialTensorCopier::PartialTensorCopier(const PartialTensorView& dst_view,
                                         const PartialTensorView& src_view,
                                         const DataType data_type)
    : dst_view_(dst_view), src_view_(src_view), data_type_(data_type) {
  CHECK_EQ(dst_view.NumAxes(), src_view.NumAxes());
  if (dst_view.Intersect(src_view).IsEmpty()) { return; }
  PartialTensorView dst_raw_view;
  PartialTensorView src_raw_view;
  const size_t size_of_data_type = GetSizeOfDataType(data_type);
  if (size_of_data_type == 1) {
    dst_raw_view = dst_view;
    src_raw_view = src_view;
  } else {
    std::vector<Range> dst_raw_range_vec = dst_view.range_vec();
    dst_raw_range_vec.back().mut_begin() = dst_raw_range_vec.back().begin() * size_of_data_type;
    dst_raw_range_vec.back().mut_end() = dst_raw_range_vec.back().end() * size_of_data_type;
    std::vector<Range> src_raw_range_vec = src_view.range_vec();
    src_raw_range_vec.back().mut_begin() = src_raw_range_vec.back().begin() * size_of_data_type;
    src_raw_range_vec.back().mut_end() = src_raw_range_vec.back().end() * size_of_data_type;
    dst_raw_view = PartialTensorView(dst_raw_range_vec);
    src_raw_view = PartialTensorView(src_raw_range_vec);
  }
  PartialTensorView folded_dst_raw_view;
  PartialTensorView folded_src_raw_view;
  PartialTensorView::FoldSameRange(dst_raw_view, src_raw_view, &folded_dst_raw_view,
                                   &folded_src_raw_view);
  PartialTensorView fold_intersection = folded_dst_raw_view.Intersect(folded_src_raw_view);
  memory_copy_nd_desc_.dst_shape = folded_dst_raw_view.shape();
  memory_copy_nd_desc_.src_shape = folded_src_raw_view.shape();
  memory_copy_nd_desc_.dst_pos = fold_intersection.OffsetTo(dst_raw_view);
  memory_copy_nd_desc_.src_pos = fold_intersection.OffsetTo(src_raw_view);
  memory_copy_nd_desc_.extent = fold_intersection.shape();
  memory_copy_nd_desc_.dst_ptr = nullptr;
  memory_copy_nd_desc_.src_ptr = nullptr;
}

void PartialTensorCopier::Exec(DeviceCtx* ctx, const MemoryCopier& copier, Blob* dst_blob,
                               Blob* src_blob) const {
  CHECK_EQ(dst_blob->data_type(), data_type_);
  CHECK_EQ(src_blob->data_type(), data_type_);
  CHECK_EQ(dst_view_.shape(), dst_blob->shape());
  CHECK_EQ(src_view_.shape(), src_blob->shape());
  memory_copy_nd_desc_.dst_ptr = dst_blob->mut_dptr();
  memory_copy_nd_desc_.src_ptr = src_blob->dptr();
  copier.Copy(ctx, memory_copy_nd_desc_);
}

}  // namespace oneflow
