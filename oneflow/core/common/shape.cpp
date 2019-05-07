#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

template<class InputIt, class OutputIt>
OutputIt CopyDims(InputIt first, InputIt last, OutputIt d_first) {
  while (first != last) {
    CHECK_GE(*first, 0LL);
    *d_first++ = *first++;
  }
  return d_first;
}

}  // namespace

Shape::Shape(const ShapeProto& shape_proto) : num_axes_(shape_proto.dim_size()) {
  CHECK_LE(num_axes_, sizeof(dim_) / sizeof(int64_t));
  CopyDims(shape_proto.dim().begin(), shape_proto.dim().end(), dim_);
  UpdateElemCnt();
}

Shape::Shape(const std::vector<int64_t>& dim_vec) : num_axes_(dim_vec.size()) {
  CHECK_LE(num_axes_, sizeof(dim_) / sizeof(int64_t));
  CopyDims(dim_vec.begin(), dim_vec.end(), dim_);
  UpdateElemCnt();
}

Shape::Shape(const std::initializer_list<int64_t>& dim_list) : num_axes_(dim_list.size()) {
  CHECK_LE(num_axes_, sizeof(dim_) / sizeof(int64_t));
  CopyDims(dim_list.begin(), dim_list.end(), dim_);
  UpdateElemCnt();
}

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  int64_t begin = ShiftNegativeAxisIfNeedAndCheck(begin_axis);
  int64_t end = ShiftNegativeAxisIfNeedAndCheck(end_axis);
  CHECK_LE(begin, end) << "Num of axes: " << num_axes_ << ", begin axis: " << begin_axis
                       << ", end axis: " << end_axis;
  int64_t elem_cnt = 1;
  FOR_RANGE(int64_t, i, begin, end) { elem_cnt *= dim_[i]; }
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
    ss << i;
    if (num_axes_ == 1 || i != num_axes_ - 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(dim_, dim_ + num_axes_);
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

}  // namespace oneflow
