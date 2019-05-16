#include "oneflow/core/register/partial_tensor_view_copier.h"

namespace oneflow {

void PartialTensorViewCopier::CopyIntersection(DeviceCtx* ctx, const PartialTensorView& dst_view,
                                               Blob* dst_blob, const PartialTensorView& src_view,
                                               const Blob* src_blob) const {
  CHECK_EQ(dst_view.size(), src_view.size());
  CHECK_EQ(dst_blob->data_type(), src_blob->data_type());
  const int64_t num_axes = dst_view.size();
  CHECK_EQ(dst_view.shape(), dst_blob->shape());
  CHECK_EQ(src_view.shape(), src_blob->shape());
  if (dst_view.Intersect(src_view).IsEmpty()) { return; }
  PartialTensorView folded_dst_view;
  PartialTensorView folded_src_view;
  PartialTensorView::JointFold(dst_view, src_view, &folded_dst_view, &folded_src_view);
  PartialTensorView fold_intersection = folded_dst_view.Intersect(folded_src_view);
}

}  // namespace oneflow
