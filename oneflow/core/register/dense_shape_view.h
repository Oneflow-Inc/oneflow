#ifndef ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_

#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class ShapeProto;
class Shape;
class PodPtr;

class DenseShapeView final {
 public:
  DenseShapeView() : ptr_(nullptr), num_axes_(0) {}
  DenseShapeView(const int64_t* ptr, int64_t num_axes) : ptr_(ptr), num_axes_(num_axes) {}
  DenseShapeView(const PodPtr& dense_shape_ptr);
  DenseShapeView(const ShapeProto& shape_proto);
  DenseShapeView(const Shape& shape);
  DenseShapeView(const DenseShapeView& rhs) = default;
  ~DenseShapeView() = default;

  int64_t NumAxes() const { return num_axes_; }
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t elem_cnt() const;
  const int64_t* ptr() const { return ptr_; }

  bool operator==(const DenseShapeView& rhs) const;
  std::string ToString() const;
  void ToDimVector(DimVector* dim_vec) const;
  void ToShape(Shape* shape) const;

 private:
  const int64_t* ptr_;
  int64_t num_axes_;
};

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape);

class DenseShapeMutView final {
 public:
  DenseShapeMutView(const PodPtr& dense_shape_ptr);
  DenseShapeMutView(const DenseShapeMutView& rhs) = default;
  ~DenseShapeMutView() = default;

  void Set(int64_t axis, int64_t val);

  void set_shape(const Shape& val);
  void set_shape(const DenseShapeView& shape);
  void LeftOnesStrippedAssign(const Shape& shape);

 private:
  int64_t* ptr_;
  int64_t num_axes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
