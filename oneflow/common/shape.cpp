#include "oneflow/common/shape.h"
#include "oneflow/common/protobuf.h"

namespace oneflow {

Shape::Shape(const ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<PbInt64> (dim_vec_.begin(), dim_vec_.end());
}

std::string Shape::DebugStr() const {
  std::stringstream ss;
  ss << "{";
  for (int64_t dim : dim_vec_) {
    ss << dim << ",";
  }
  ss << "(" << elem_cnt_ << ")}";
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
  if (dim_vec_.size() == 0) {
    elem_cnt_ = 0;
  }
}

std::ostream& operator<< (std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

} // namespace oneflow
