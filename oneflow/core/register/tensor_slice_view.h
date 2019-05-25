#ifndef ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/index.h"
#include "oneflow/core/register/tensor_slice_view.pb.h"

namespace oneflow {

class TensorSliceView final {
 public:
  TensorSliceView() = default;
  TensorSliceView(const std::initializer_list<Range>& ranges);
  explicit TensorSliceView(const std::vector<Range>& ranges);
  explicit TensorSliceView(const TensorSliceViewProto& proto);

  TensorSliceView& operator=(const TensorSliceView& other);
  bool operator==(const TensorSliceView& rhs) const;
  bool operator!=(const TensorSliceView& rhs) const;

  bool IsEmpty() const;
  TensorSliceView Intersect(const TensorSliceView& other) const;
  const Range& At(int64_t index) const;
  const Shape& shape() const;
  const std::vector<Range>& range_vec() const;
  size_t NumAxes() const;
  Index OffsetTo(const TensorSliceView& other) const;
  void ToProto(TensorSliceViewProto* proto) const;

 private:
  std::vector<Range> range_vec_;
  Shape shape_;

  void UpdateShape();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_
