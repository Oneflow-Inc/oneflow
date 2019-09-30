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
  dim_vec_.at(index) = val;
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
    if (axis_vec.at(i) < 0) { ret.at(i) += num_axes; }
    CHECK_LT(ret.at(i), num_axes);
    CHECK_GE(ret.at(i), 0);
  }
  return ret;
}

Shape Shape::CreateReducedShape(const std::vector<int64_t>& axis_vec) const {
  CHECK_EQ(axis_vec.empty(), false);
  std::vector<int64_t> dim_vec = this->dim_vec();
  for (const int64_t& axis : ShiftNegativeAxis(axis_vec)) { dim_vec.at(axis) = 1; }
  return Shape(dim_vec);
}

Shape Shape::RemoveOnes(const std::vector<int64_t>& axis_vec) const {
  std::vector<int64_t> dim_vec;
  const std::vector<int64_t> axis_vec_shifted = ShiftNegativeAxis(axis_vec);
  for (int64_t i = 0; i < this->dim_vec().size(); i++) {
    if (std::find(axis_vec_shifted.begin(), axis_vec_shifted.end(), i) == axis_vec_shifted.end()) {
      dim_vec.push_back(this->dim_vec().at(i));
    } else {
      CHECK_EQ(this->dim_vec().at(i), 1);
    }
  }
  if (dim_vec.empty()) { dim_vec.push_back(1); }
  return Shape(dim_vec);
}

Shape Shape::Ones(const int64_t num_axes) {
  std::vector<int64_t> dim_vec(num_axes);
  std::fill(dim_vec.begin(), dim_vec.end(), 1);
  return Shape(dim_vec);
}

Shape Shape::CreateReducedShapeOrOnesShape(const std::vector<int64_t>& axis_vec) const {
  if (axis_vec.empty()) { return Shape::Ones(NumAxes()); }
  return CreateReducedShape(axis_vec);
}

std::vector<int64_t> Shape::Axes4BroadcastTo(const Shape& broadcast_shape) const {
  std::vector<int64_t> broadcast_axis_vec;
  CHECK_EQ(broadcast_shape.NumAxes(), NumAxes());
  for (int64_t i = 0; i < NumAxes(); i++) {
    if (this->dim_vec().at(i) != broadcast_shape.dim_vec().at(i) && this->dim_vec().at(i) == 1) {
      broadcast_axis_vec.push_back(i);
    } else {
      CHECK_EQ(this->dim_vec().at(i), broadcast_shape.dim_vec().at(i));
    }
  }
  CHECK(!broadcast_axis_vec.empty());
  return broadcast_axis_vec;
}

bool Shape::Containing(const Shape& small_shape) const {
  if (this->NumAxes() < small_shape.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, small_shape.NumAxes()) {
    if (this->At(i) != small_shape.At(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow
