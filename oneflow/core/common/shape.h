#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/util.h"

#include <cstring>

namespace oneflow {

class Shape final {
 public:
  Shape() : num_axes_(0), elem_cnt_(1) {}
  explicit Shape(const ShapeProto& shape_proto);
  explicit Shape(const std::vector<int64_t>& dim_vec);
  Shape(const std::initializer_list<int64_t>& dim_vec);
  ~Shape() = default;

  bool operator==(const Shape& rhs) const {
    return num_axes_ == rhs.num_axes() && std::memcmp(dim_, rhs.dim(), GetElemsTotalBytesSize());
  }

  bool operator!=(const Shape& rhs) const {
    return num_axes_ != rhs.num_axes() || !std::memcmp(dim_, rhs.dim(), GetElemsTotalBytesSize());
  }

  void Set(int64_t axis, int64_t dim) {
    int64_t raxis = ShiftNegativeAxisIfNeedAndCheck(axis);
    dim_[raxis] = dim;
    UpdateElemCnt();
  }
  int64_t At(int64_t axis) const {
    int64_t raxis = ShiftNegativeAxisIfNeedAndCheck(axis);
    return dim_[raxis];
  }
  int64_t NumAxes() const { return num_axes_; }
  int64_t Count() const { return elem_cnt(); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t CountFrom(int64_t begin_axis) const { return Count(begin_axis, num_axes()); }
  int64_t CountTo(int64_t end_axis) const { return Count(0, end_axis); }
  // deprecated
  int64_t Count(int64_t begin_axis) const { return Count(begin_axis, num_axes()); }

  std::vector<int64_t> dim_vec() const { return std::vector<int64_t>(dim_, dim_ + num_axes_); }
  Shape CreateLeftExtendedShape(size_t extend_axes) const;

  std::string ToString() const;
  std::string DebugStr() const { return ToString(); }
  void ToProto(ShapeProto*) const;
  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  size_t num_axes() const { return num_axes_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  const int64_t* dim() const { return dim_; }
  int64_t* mut_dim() { return dim_; }

 private:
  size_t GetElemsTotalBytesSize() const { return elem_cnt_ * sizeof(int64_t); }
  void UpdateElemCnt() {
    elem_cnt_ = 1;
    FOR_RANGE(size_t, i, 0, num_axes_) { elem_cnt_ *= dim_[i]; }
  }
  int64_t ShiftNegativeAxisIfNeedAndCheck(int64_t axis) const {
    int64_t regular_axis = axis < 0 ? num_axes_ + axis : axis;
    CHECK_GE(regular_axis, 0);
    CHECK_LT(regular_axis, num_axes_);
    return regular_axis;
  }

 private:
  size_t num_axes_;
  int64_t elem_cnt_;
  int64_t dim_[OF_PP_SEQ_SIZE(DIM_SEQ)];
};

template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  FOR_RANGE(int64_t, i, 0, num_axes_) { out_stream << std::to_string(dim_[i]) << ' '; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
