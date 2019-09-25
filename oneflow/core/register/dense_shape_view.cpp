#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

DenseShapeViewBase::DenseShapeViewBase(PodPtr dense_shape_ptr) {
  ptr_ = dense_shape_ptr.MutTensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  const TensorPodDesc& dense_shape_desc = dense_shape_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
  num_axes_ = dense_shape_desc.shape().At(0);
}

DenseShapeView::operator Shape() const {
  std::vector<int64_t> dim_vec;
  FOR_RANGE(int, i, 0, NumAxes()) { dim_vec.push_back(At(i)); }
  return Shape(dim_vec);
}

bool DenseShapeView::operator==(const DenseShapeView& rhs) const {
  if (NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, NumAxes()) {
    if (At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

int64_t DenseShapeView::At(int64_t index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_axes_);
  return ptr_[index];
}

int64_t DenseShapeView::Count(int64_t begin_axis) const { return Count(begin_axis, NumAxes()); }

int64_t DenseShapeView::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}

int64_t DenseShapeView::elem_cnt() const { return Count(0); }

std::string DenseShapeView::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(int, i, 0, NumAxes()) {
    int64_t dim = At(i);
    ss << dim;
    if (i != NumAxes() - 1 || NumAxes() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape) {
  out << shape.ToString();
  return out;
}

void DenseShapeMutView::set_shape(const Shape& shape) {
  CHECK_EQ(num_axes_, shape.NumAxes());
  for (size_t i = 0; i < shape.NumAxes(); ++i) { ptr_[i] = shape.At(i); }
}

}  // namespace oneflow
