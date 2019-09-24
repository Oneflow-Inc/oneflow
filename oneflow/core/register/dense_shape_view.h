#ifndef ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

class DenseShapeViewBase {
 protected:
  DenseShapeViewBase(PodPtr dense_shape_ptr);
  DenseShapeViewBase(const DenseShapeViewBase& rhs) = default;
  virtual ~DenseShapeViewBase() = default;

  int64_t* ptr_;
  int64_t num_axes_;
};

class DenseShapeView final : public DenseShapeViewBase {
 public:
  DenseShapeView(const PodPtr& dense_shape_ptr) : DenseShapeViewBase(dense_shape_ptr) {}
  DenseShapeView(const DenseShapeView& rhs) : DenseShapeViewBase(rhs) {}
  ~DenseShapeView() = default;

  int64_t NumAxes() const { return num_axes_; }
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t elem_cnt() const;

  bool operator==(const DenseShapeView& rhs) const;
  std::string ToString() const;

  operator Shape() const;
};

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape);

class DenseShapeMutView final : public DenseShapeViewBase {
 public:
  DenseShapeMutView(PodPtr dense_shape_ptr) : DenseShapeViewBase(dense_shape_ptr) {}
  DenseShapeMutView(const DenseShapeView& rhs) : DenseShapeViewBase(rhs) {}
  ~DenseShapeMutView() = default;

  void set_shape(const Shape& val);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_DENSE_SHAPE_VIEW_H_
