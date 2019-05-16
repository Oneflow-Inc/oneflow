#include "oneflow/core/register/tensor_copy_desc.h"

namespace oneflow {

TensorCopyDesc::TensorCopyDesc(void *dst_ptr, const void *src_ptr, const Shape &dst_shape,
                               const Shape &src_shape, const std::vector<int64_t> &dst_pos,
                               const std::vector<int64_t> &src_pos, const Shape &extent)
    : dst_ptr_(dst_ptr),
      src_ptr_(src_ptr),
      dst_shape_(dst_shape),
      src_shape_(src_shape),
      dst_pos_(dst_pos),
      src_pos_(src_pos),
      extent_(extent) {
  num_axes_ = dst_shape_.NumAxes();
  CHECK_EQ(src_shape_.NumAxes(), num_axes_);
  CHECK_EQ(dst_pos_.size(), num_axes_);
  CHECK_EQ(src_pos_.size(), num_axes_);
  CHECK_EQ(extent_.NumAxes(), num_axes_);
}

}  // namespace oneflow
