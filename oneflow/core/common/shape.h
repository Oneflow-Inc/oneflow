#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Shape final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  Shape() : elem_cnt_(0) {}
  explicit Shape(const std::vector<int64_t>& dim_vec);
  explicit Shape(const ShapeProto& shape_proto);
  Shape(const std::initializer_list<int64_t>& dim_vec);
  ~Shape() = default;
  Shape& operator=(const Shape& shape);

  bool operator==(const Shape& rhs) const;
  bool operator!=(const Shape& rhs) const { return !(*this == rhs); }
  std::string DebugStr() const;
  std::string ToString() const;

  void ToProto(ShapeProto*) const;

  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  // Getters and Setters
  const std::vector<int64_t>& dim_vec() const { return dim_vec_; }
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t At(int64_t index) const { return dim_vec_[index]; }
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const { return dim_vec_.size(); }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;

  Shape CreateLeftExtendedShape(int num_axes) const;
  std::vector<int64_t> ShiftNegativeAxis(const std::vector<int64_t>& axis_vec) const;
  Shape CreateReducedShape(const std::vector<int64_t>& axis_vec) const;
  Shape CreateReducedShapeOrOnesShape(const std::vector<int64_t>& axis_vec) const;
  Shape RemoveOnes(const std::vector<int64_t>& axis_vec) const;
  static Shape Ones(const int64_t num_axes);
  std::vector<int64_t> Axes4BroadcastTo(const Shape& broadcast_dim_vec) const;

  bool Containing(const Shape& small_shape) const;

 private:
  void UpdateElemCnt();

  std::vector<int64_t> dim_vec_;
  int64_t elem_cnt_;
};

template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  for (int64_t dim : dim_vec_) { out_stream << std::to_string(dim) << ' '; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Shape> {
  size_t operator()(const oneflow::Shape& shape) const {
    size_t ret = 0;
    FOR_RANGE(int, i, 0, shape.NumAxes()) { ret ^= std::hash<int64_t>()(shape.At(i)); }
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
