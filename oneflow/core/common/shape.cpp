#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  int64_t begin = ShiftNegativeAxisIfNeedAndCheck(begin_axis);
  int64_t end = ShiftNegativeAxisIfNeedAndCheck(end_axis);
  CHECK_LE(begin, end) << "Num of axes: " << num_axes_ << ", begin axis: " << begin_axis
                       << ", end axis: " << end_axis;
  int64_t elem_cnt = 1;
  FOR_RANGE(int64_t, i, begin, end) { elem_cnt *= dim_array_.at(i); }
  return elem_cnt;
}

Shape Shape::CreateLeftExtendedShape(size_t extend_axes) const {
  CHECK_GE(extend_axes, num_axes_);
  std::vector<int64_t> dim_vec = this->dim_vec();
  FOR_RANGE(size_t, i, 0, extend_axes - num_axes_) { dim_vec.insert(dim_vec.begin(), 1LL); }
  return Shape(dim_vec);
}

std::string Shape::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(size_t, i, 0, num_axes_) {
    ss << dim_array_.at(i);
    if (num_axes_ == 1 || i != num_axes_ - 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = {dim_array_.begin(), dim_array_.begin() + num_axes_};
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

}  // namespace oneflow
