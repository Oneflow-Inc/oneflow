#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Shape final {
  const static size_t kMaxNumAxes = 12;
  using DimArray = std::array<int64_t, kMaxNumAxes>;

 public:
  Shape() : num_axes_(0) {}
  explicit Shape(const ShapeProto& shape_proto) { Init(shape_proto.dim()); }
  explicit Shape(const std::vector<int64_t>& dim_vec) { Init(dim_vec); }
  Shape(const std::initializer_list<int64_t>& ilist) { Init(ilist); }
  ~Shape() = default;

  bool operator==(const Shape& rhs) const {
    return num_axes_ == rhs.num_axes() 
           && std::equal(dim_array_.begin(), dim_array_.begin() + num_axes_,	
                         rhs.dim_array().begin());;
  }

  bool operator!=(const Shape& rhs) const {
    return !(*this == rhs);
  }

  int64_t NumAxes() const { return num_axes_; }
  int64_t ShiftIfNeed(int64_t axis) const { return axis < 0 ? num_axes_ + axis : axis; }
  void Set(int64_t axis, int64_t dim) {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, num_axes_);
    dim_array_[axis] = dim;
  }
  int64_t At(int64_t axis) const {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, num_axes_);
    return dim_array_[axis];
  }
  int64_t Count() const { return Count(0, NumAxes()); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t CountFrom(int64_t begin_axis) const { return Count(begin_axis, num_axes()); }
  int64_t CountTo(int64_t end_axis) const { return Count(0, end_axis); }
  // deprecated, use CountFrom instead
  int64_t Count(int64_t begin_axis) const { return CountFrom(begin_axis); }
  // deprecated, use Count instead
  int64_t elem_cnt() const { return Count(); }

  std::vector<int64_t> dim_vec() const {
    return std::vector<int64_t>(dim_array_.begin(), dim_array_.begin() + num_axes_);
  }
  Shape CreateLeftExtendedShape(size_t extend_axes) const;

  std::string ToString() const;
  std::string DebugStr() const { return ToString(); }
  void ToProto(ShapeProto*) const;
  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  size_t num_axes() const { return num_axes_; }
  const DimArray& dim_array() const { return dim_array_; }
  DimArray& mut_dim_array() { return dim_array_; }

 private:
  template<typename T>
  void Init(const T& other);

 private:
  size_t num_axes_;
  DimArray dim_array_;
};

template<typename T>
void Shape::Init(const T& other) {
  num_axes_ = other.size();
  CHECK_LE(num_axes_, dim_array_.size());
  for (auto it = other.begin(); it != other.end(); ++it) { CHECK_GE(*it, 0LL); }
  std::copy(other.begin(), other.end(), dim_array_.begin());
}

template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  FOR_RANGE(int64_t, i, 0, num_axes_) { out_stream << std::to_string(dim_array_[i]) << ' '; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
