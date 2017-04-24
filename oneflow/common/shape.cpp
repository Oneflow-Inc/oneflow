#include "common/shape.h"

namespace oneflow {

std::string Shape::ToString() const {
  std::ostringstream oss;
  for (int64_t dim : shape_vec_) {
    oss << dim << " ";
  }
  oss << "(" << elem_cnt_ << ")";
  return oss.str();
}

int64_t Shape::Count(int64_t start_axis, int64_t end_axis) const {
  CHECK(0 <= start_axis && start_axis <= end_axis && end_axis <= NumAxes())
      << "[start_axis:" << start_axis
      << "][end_axis:" << end_axis
      << "][num_axes:" << NumAxes() << "]";
  int64_t cnt = 1;
  for (int64_t i = start_axis; i < end_axis; ++i) {
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
  for (int64_t s : shape_vec_) {
    elem_cnt_ *= s;
  }
}

} // namespace oneflow
