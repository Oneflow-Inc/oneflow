#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Shape::Shape(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }
Shape::Shape(const std::vector<int64_t>& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }

Shape::Shape(const ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

Shape& Shape::operator=(const Shape& shape) {
  dim_vec_ = shape.dim_vec_;
  UpdateElemCnt();
  return *this;
}

bool Shape::operator==(const Shape& rhs) const { return dim_vec_ == rhs.dim_vec_; }

std::string Shape::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : dim_vec_) {
    ss << dim;
    if (++idx != dim_vec_.size() || dim_vec_.size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

std::string Shape::DebugStr() const { return ToString(); }

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(dim_vec_.begin(), dim_vec_.end());
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

int64_t Shape::Count(int64_t begin_axis) const { return Count(begin_axis, NumAxes()); }

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (int64_t s : dim_vec_) { elem_cnt_ *= s; }
  if (dim_vec_.size() == 0) { elem_cnt_ = 0; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

Shape Shape::CreateLeftExtendedShape(int num_axes) const {
  CHECK_GE(num_axes, NumAxes());
  std::vector<int64_t> dim_vec = this->dim_vec();
  for (int i = 0; i < num_axes - NumAxes(); ++i) { dim_vec.insert(dim_vec.begin(), 1LL); }
  return Shape(dim_vec);
}

std::vector<int64_t> Shape::ShiftNegativeAxis(const std::vector<int64_t>& axis_vec) const {
  const int64_t num_axes = this->NumAxes();
  std::vector<int64_t> ret = axis_vec;
  for (int64_t i = 0; i < axis_vec.size(); i++) {
    if (axis_vec[i] < 0) { ret[i] += num_axes; }
    CHECK_LT(ret[i], num_axes);
    CHECK_GE(ret[i], 0);
  }
  return ret;
}

Shape Shape::CreateReducedShape(const std::vector<int64_t>& axis_vec, bool keep_dims) const {
  if (keep_dims) {
    std::vector<int64_t> dim_vec = ShiftNegativeAxis(axis_vec);
    for (const int64_t& axis : axis_vec) { dim_vec[axis] = 1; }
    return Shape(dim_vec);
  } else {
    std::vector<int64_t> dim_vec;
    std::vector<int64_t> dim2is_reduced(this->dim_vec().size());
    for (const auto& axis : axis_vec) { dim2is_reduced[axis] = 1; }
    FOR_RANGE(int64_t, i, 0, this->NumAxes()) {
      if (dim2is_reduced[i] != 1) { dim_vec.push_back(this->dim_vec()[i]); }
    }
    if (dim_vec.empty()) { dim_vec.push_back(1); }
    return Shape(dim_vec);
  }
}

}  // namespace oneflow
