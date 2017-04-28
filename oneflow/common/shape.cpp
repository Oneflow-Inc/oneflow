#include "common/shape.h"

namespace oneflow {

Shape::Shape(const ShapeProto& shape_proto) {
  TODO();
}

ShapeProto Shape::ToProto() const {
  TODO();
}

std::string Shape::ToString() const {
  std::stringstream ss;
  for (int64_t dim : dim_vec_) {
    ss << dim << " ";
  }
  ss << "(" << elem_cnt_ << ")";
  return ss.str();
}

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << "[begin_axis:" << begin_axis
      << "][end_axis:" << end_axis
      << "][num_axes:" << NumAxes() << "]";
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) {
    cnt *= At(i);
  }
  return cnt;
}

int64_t Shape::CanonicalAxisIndex(int64_t axis_index) const {
  CHECK_GE(axis_index, -NumAxes());
  CHECK_LT(axis_index, NumAxes());
  return (axis_index + NumAxes()) % NumAxes();
}

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (int64_t s : dim_vec_) {
    elem_cnt_ *= s;
  }
}

} // namespace oneflow
