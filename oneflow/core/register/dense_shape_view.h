#ifndef ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_

#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class ShapeProto;
class Shape;

template<typename DimT>
class DenseShapeViewBase {
 public:
  using DimType = DimT;
  DenseShapeViewBase(DimType* ptr, int64_t num_axes) : ptr_(ptr), num_axes_(num_axes) {}
  DenseShapeViewBase(const DenseShapeViewBase& rhs) = default;
  ~DenseShapeViewBase() = default;

  int64_t NumAxes() const { return num_axes_; }
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t elem_cnt() const;
  const DimType* ptr() const { return ptr_; }

  bool operator==(const DenseShapeViewBase& rhs) const;
  std::string ToString() const;
  void ToDimVector(DimVector* dim_vec) const;
  void ToShape(Shape* shape) const;

  void set_ptr(DimType* ptr) { ptr_ = ptr; }

 protected:
  DimType* dim_ptr() const { return ptr_; }

 private:
  DimType* ptr_;
  int64_t num_axes_;
};

class DenseShapeView final : public DenseShapeViewBase<const int64_t> {
 public:
  DenseShapeView() : DenseShapeViewBase<const int64_t>(nullptr, 0) {}
  DenseShapeView(const int64_t* ptr, int64_t num_axes)
      : DenseShapeViewBase<const int64_t>(ptr, num_axes) {}
  DenseShapeView(const ShapeProto& shape_proto);
  DenseShapeView(const Shape& shape);
  DenseShapeView(const DenseShapeView& rhs) = default;
  ~DenseShapeView() = default;
};

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape);

class DenseShapeMutView final : public DenseShapeViewBase<int64_t> {
 public:
  DenseShapeMutView() : DenseShapeViewBase<int64_t>(nullptr, 0) {}
  DenseShapeMutView(int64_t* ptr, int64_t num_axes) : DenseShapeViewBase<int64_t>(ptr, num_axes) {}
  DenseShapeMutView(const DenseShapeMutView& rhs) = default;
  ~DenseShapeMutView() = default;

  int64_t* mut_ptr() const { return dim_ptr(); }
  void Set(int64_t axis, int64_t val);

  void set_shape(const Shape& val);
  void set_shape(const DenseShapeView& shape);
};

template<typename DimT>
bool DenseShapeViewBase<DimT>::operator==(const DenseShapeViewBase<DimT>& rhs) const {
  if (this->NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    if (At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
