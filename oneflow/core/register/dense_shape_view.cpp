#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

DenseShapeView::DenseShapeView(const PodPtr& dense_shape_ptr) {
  ptr_ = dense_shape_ptr.TensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  const TensorPodDesc& dense_shape_desc = dense_shape_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
  num_axes_ = dense_shape_desc.shape().At(0);
}

DenseShapeView::DenseShapeView(const ShapeProto& shape_proto) {
  ptr_ = shape_proto.dim().data();
  num_axes_ = shape_proto.dim_size();
}

DenseShapeView::DenseShapeView(const Shape& shape) {
  ptr_ = shape.dim_vec().data();
  num_axes_ = shape.dim_vec().size();
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

void DenseShapeView::ToDimVector(DimVector* dim_vec) const {
  dim_vec->resize(num_axes_);
  dim_vec->assign(ptr_, ptr_ + num_axes_);
}

void DenseShapeView::ToShape(Shape* shape) const {
  DimVector dim_vec;
  ToDimVector(&dim_vec);
  *shape = Shape(std::move(dim_vec));
}

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape) {
  out << shape.ToString();
  return out;
}

DenseShapeMutView::DenseShapeMutView(const PodPtr& dense_shape_ptr) {
  ptr_ = PodPtr(dense_shape_ptr).MutTensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  const TensorPodDesc& dense_shape_desc = dense_shape_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
  num_axes_ = dense_shape_desc.shape().At(0);
}

void DenseShapeMutView::Set(int64_t axis, int64_t val) {
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes_);
  ptr_[axis] = val;
}

void DenseShapeMutView::set_shape(const Shape& shape) {
  CHECK_EQ(num_axes_, shape.NumAxes());
  std::copy(shape.dim_vec().data(), shape.dim_vec().data() + shape.NumAxes(), ptr_);
}

void DenseShapeMutView::set_shape(const DenseShapeView& shape) {
  CHECK_EQ(num_axes_, shape.NumAxes());
  std::copy(shape.ptr(), shape.ptr() + shape.NumAxes(), ptr_);
}

void DenseShapeMutView::LeftOnesStrippedAssign(const Shape& shape) {
  CHECK_LE(num_axes_, shape.NumAxes());
  size_t left_ones_len = shape.NumAxes() - num_axes_;
  FOR_RANGE(int, i, 0, left_ones_len) { CHECK_EQ(shape.At(i), 1LL); }
  const int64_t* const_ptr = shape.dim_vec().data() + left_ones_len;
  std::copy(const_ptr, const_ptr + num_axes_, ptr_);
}

}  // namespace oneflow
