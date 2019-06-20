#include "oneflow/core/register/tensor_slice_copier.h"

namespace oneflow {

namespace {

TensorSliceView GetRawTenserSliceView(const TensorSliceView& view, DataType data_type) {
  const size_t size_of_data_type = GetSizeOfDataType(data_type);
  if (size_of_data_type == 1) {
    return view;
  } else {
    std::vector<Range> range_vec = view.range_vec();
    range_vec.back().mut_begin() = range_vec.back().begin() * size_of_data_type;
    range_vec.back().mut_end() = range_vec.back().end() * size_of_data_type;
    return TensorSliceView(range_vec);
  }
}

}  // namespace

TensorSliceCopier::TensorSliceCopier(const TensorSliceView& dst_view,
                                     const TensorSliceView& src_view,
                                     const TensorSliceView& copy_view, const DataType data_type)
    : dst_view_(dst_view), src_view_(src_view), data_type_(data_type) {
  CHECK(dst_view.Contains(copy_view));
  CHECK(src_view.Contains(copy_view));
  TensorSliceView dst_raw_view = GetRawTenserSliceView(dst_view, data_type);
  TensorSliceView src_raw_view = GetRawTenserSliceView(src_view, data_type);
  TensorSliceView copy_raw_view = GetRawTenserSliceView(copy_view, data_type);
  MemoryCopyNdDesc raw_copy_desc;
  raw_copy_desc.dst_shape = dst_raw_view.shape();
  raw_copy_desc.src_shape = src_raw_view.shape();
  raw_copy_desc.dst_pos = copy_raw_view.OffsetTo(dst_raw_view);
  raw_copy_desc.src_pos = copy_raw_view.OffsetTo(src_raw_view);
  raw_copy_desc.extent = copy_raw_view.shape();
  raw_copy_desc.dst_ptr = nullptr;
  raw_copy_desc.src_ptr = nullptr;
  memory_copy_nd_desc_ = raw_copy_desc.CreateDimReducedDesc();
}

TensorSliceCopier::TensorSliceCopier(const TensorSliceView& dst_view,
                                     const TensorSliceView& src_view, const DataType data_type)
    : TensorSliceCopier(dst_view, src_view, dst_view.Intersect(src_view), data_type) {}

void TensorSliceCopier::Copy(DeviceCtx* ctx, const MemoryCopier& copier, Blob* dst_blob,
                             const Blob* src_blob) const {
  CHECK_EQ(dst_blob->data_type(), data_type_);
  CHECK_EQ(src_blob->data_type(), data_type_);
  CHECK_EQ(dst_view_.shape().elem_cnt(), dst_blob->shape().elem_cnt());
  CHECK_EQ(src_view_.shape().elem_cnt(), src_blob->shape().elem_cnt());
  memory_copy_nd_desc_.dst_ptr = dst_blob->mut_dptr();
  memory_copy_nd_desc_.src_ptr = src_blob->dptr();
  copier.Copy(ctx, memory_copy_nd_desc_);
}

}  // namespace oneflow
