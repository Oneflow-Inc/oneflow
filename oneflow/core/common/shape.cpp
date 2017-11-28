#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Shape::Shape(const std::vector<int64_t>& dim_vec) : dim_vec_(dim_vec) {
  UpdateElemCnt();
}

Shape::Shape(const ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

bool Shape::operator==(const Shape& rhs) const {
  return dim_vec_ == rhs.dim_vec_;
}

std::string Shape::DebugStr() const {
  std::stringstream ss;
  ss << "{";
  for (int64_t dim : dim_vec_) { ss << dim << ","; }
  ss << "(" << elem_cnt_ << ")}";
  return ss.str();
}

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<PbInt64>(dim_vec_.begin(), dim_vec_.end());
}

void Shape::Set(int64_t index, int64_t val) {
  dim_vec_[index] = val;
  UpdateElemCnt();
}

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}

int64_t Shape::Count(int64_t begin_axis) const {
  return Count(begin_axis, NumAxes());
}

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (int64_t s : dim_vec_) { elem_cnt_ *= s; }
  if (dim_vec_.size() == 0) { elem_cnt_ = 0; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

}  // namespace oneflow
