#include "oneflow/core/register/partial_tensor_view_copier.h"

namespace oneflow {

namespace {

void CheckViewBlob(const PartialTensorView& view, const Blob* blob) {
  const Shape& shape = blob->shape();
  CHECK_EQ(shape.NumAxes(), view.dim_size());
  FOR_RANGE(int32_t, i, 0, view.dim_size()) {
    const RangeProto& range = view.dim(i);
    CHECK_GE(range.begin(), 0);
    CHECK_GT(range.end(), range.begin());
    CHECK_EQ(range.end() - range.begin(), shape.At(i));
  }
}

}  // namespace

void PartialTensorViewCopier::CopyOverlap(DeviceCtx* ctx, const PartialTensorView& dst_view,
                                          Blob* dst_blob, const PartialTensorView& src_view,
                                          const Blob* src_blob) const {
  CHECK_EQ(dst_view.dim_size(), src_view.dim_size());
  CHECK_EQ(dst_blob->data_type(), src_blob->data_type());
  const int64_t num_axes = dst_view.dim_size();
  CheckViewBlob(dst_view, dst_blob);
  CheckViewBlob(src_view, src_blob);
  PartialTensorView overlap_view;
  std::vector<int64_t> diff_axes;
  FOR_RANGE(int32_t, i, 0, num_axes) {
    const RangeProto& dst_range = dst_view.dim(i);
    const RangeProto& src_range = src_view.dim(i);
    if (src_range.begin() >= dst_range.end() || dst_range.begin() >= src_range.end()) {
      return;
    } else if (src_range.begin() == dst_range.begin() && src_range.end() == dst_range.end()) {
      *overlap_view.add_dim() = src_range;
    } else {
      RangeProto* overlap_range = overlap_view.add_dim();
      overlap_range->set_begin(std::max(src_range.begin(), dst_range.begin()));
      overlap_range->set_end(std::min(src_range.end(), dst_range.end()));
      diff_axes.push_back(i);
    }
  }
}

}  // namespace oneflow
