#ifndef ONEFLOW_CORE_REGISTER_TENSOR_COPY_DESC_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_COPY_DESC_H_

#include "oneflow/core/common/shape.h"

namespace oneflow {

class TensorCopyDesc final {
 public:
  TensorCopyDesc(void *dst_ptr, const void *src_ptr, const Shape &dst_shape, const Shape &src_shape,
                 const std::vector<int64_t> &dst_pos, const std::vector<int64_t> &src_pos,
                 const Shape &extent);

  int64_t NumAxes() const { return num_axes_; };
  void *dst_ptr() const { return dst_ptr_; };
  const void *src_ptr() const { return src_ptr_; }
  const Shape &dst_shape() const { return dst_shape_; };
  const Shape &src_shape() const { return src_shape_; };
  const std::vector<int64_t> &dst_pos() const { return dst_pos_; };
  const std::vector<int64_t> &src_pos() const { return src_pos_; };
  const Shape &extent() const { return extent_; };

 private:
  int64_t num_axes_;
  void *dst_ptr_;
  const void *src_ptr_;
  Shape dst_shape_;
  Shape src_shape_;
  std::vector<int64_t> dst_pos_;
  std::vector<int64_t> src_pos_;
  Shape extent_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_COPY_DESC_H_
