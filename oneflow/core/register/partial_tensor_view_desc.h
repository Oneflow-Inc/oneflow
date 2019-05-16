#ifndef ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_DESC_H_
#define ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_DESC_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class PartialTensorViewDesc final {
 public:
  PartialTensorViewDesc() = default;
  PartialTensorViewDesc(const std::initializer_list<Range>& ranges);
  explicit PartialTensorViewDesc(const std::vector<Range>& ranges);

  PartialTensorViewDesc& operator=(const PartialTensorViewDesc& other);
  bool operator==(const PartialTensorViewDesc& rhs) const;
  bool operator!=(const PartialTensorViewDesc& rhs) const;

  bool IsEmpty() const;
  PartialTensorViewDesc Intersect(const PartialTensorViewDesc& other) const;
  const Range& At(int64_t index) const;
  const Shape& shape() const;
  size_t size() const;

  static void JointFold(const PartialTensorViewDesc& lhs, const PartialTensorViewDesc& rhs,
                        PartialTensorViewDesc* lhs_out, PartialTensorViewDesc* rhs_out);

 private:
  std::vector<Range> range_vec_;
  Shape shape_;

  void UpdateShape();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_DESC_H_
