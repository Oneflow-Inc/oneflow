#ifndef ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_VIEW_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_VIEW_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/index.h"
#include "oneflow/core/register/tensor_partial_view.pb.h"

namespace oneflow {

class TensorPartialView final {
 public:
  TensorPartialView() = default;
  TensorPartialView(const std::initializer_list<Range>& ranges);
  explicit TensorPartialView(const std::vector<Range>& ranges);
  explicit TensorPartialView(const TensorPartialViewProto& proto);

  TensorPartialView& operator=(const TensorPartialView& other);
  bool operator==(const TensorPartialView& rhs) const;
  bool operator!=(const TensorPartialView& rhs) const;

  bool IsEmpty() const;
  TensorPartialView Intersect(const TensorPartialView& other) const;
  const Range& At(int64_t index) const;
  const Shape& shape() const;
  const std::vector<Range>& range_vec() const;
  size_t NumAxes() const;
  Index OffsetTo(const TensorPartialView& other) const;
  void ToProto(TensorPartialViewProto* proto) const;

 private:
  std::vector<Range> range_vec_;
  Shape shape_;

  void UpdateShape();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_VIEW_H_
