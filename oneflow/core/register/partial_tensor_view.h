#ifndef ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_H_
#define ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/index.h"

namespace oneflow {

class PartialTensorView final {
 public:
  PartialTensorView() = default;
  PartialTensorView(const std::initializer_list<Range>& ranges);
  explicit PartialTensorView(const std::vector<Range>& ranges);

  PartialTensorView& operator=(const PartialTensorView& other);
  bool operator==(const PartialTensorView& rhs) const;
  bool operator!=(const PartialTensorView& rhs) const;

  bool IsEmpty() const;
  PartialTensorView Intersect(const PartialTensorView& other) const;
  const Range& At(int64_t index) const;
  const Shape& shape() const;
  const std::vector<Range>& range_vec() const;
  size_t NumAxes() const;
  Index OffsetTo(const PartialTensorView& other);

  static void FoldSameRange(const PartialTensorView& lhs, const PartialTensorView& rhs,
                            PartialTensorView* lhs_out, PartialTensorView* rhs_out);

 private:
  std::vector<Range> range_vec_;
  Shape shape_;

  void UpdateShape();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_H_
