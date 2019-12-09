#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

DenseShapeView::DenseShapeView(const ShapeProto& shape_proto)
    : DenseShapeViewBase<const int64_t>(shape_proto.dim().data(), shape_proto.dim_size()) {}
DenseShapeView::DenseShapeView(const Shape& shape)
    : DenseShapeViewBase<const int64_t>(shape.dim_vec().data(), shape.dim_vec().size()) {}

template<typename DimT>
int64_t DenseShapeViewBase<DimT>::At(int64_t index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_axes_);
  return ptr_[index];
}

template<typename DimT>
int64_t DenseShapeViewBase<DimT>::Count(int64_t begin_axis) const {
  return this->Count(begin_axis, NumAxes());
}

template<typename DimT>
int64_t DenseShapeViewBase<DimT>::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= this->NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= this->At(i); }
  return cnt;
}

template<typename DimT>
int64_t DenseShapeViewBase<DimT>::elem_cnt() const {
  return this->Count(0);
}

template<typename DimT>
std::string DenseShapeViewBase<DimT>::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    int64_t dim = this->At(i);
    ss << dim;
    if (i != this->NumAxes() - 1 || this->NumAxes() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

template<typename DimT>
void DenseShapeViewBase<DimT>::ToDimVector(DimVector* dim_vec) const {
  dim_vec->resize(num_axes_);
  dim_vec->assign(ptr_, ptr_ + num_axes_);
}

template<typename DimT>
void DenseShapeViewBase<DimT>::ToShape(Shape* shape) const {
  DimVector dim_vec;
  this->ToDimVector(&dim_vec);
  *shape = Shape(std::move(dim_vec));
}

template class DenseShapeViewBase<const int64_t>;
template class DenseShapeViewBase<int64_t>;

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape) {
  out << shape.ToString();
  return out;
}

void DenseShapeMutView::Set(int64_t axis, int64_t val) {
  CHECK_GE(axis, 0);
  CHECK_LT(axis, NumAxes());
  dim_ptr()[axis] = val;
}

void DenseShapeMutView::set_shape(const Shape& shape) {
  CHECK_EQ(NumAxes(), shape.NumAxes());
  std::copy(shape.dim_vec().data(), shape.dim_vec().data() + shape.NumAxes(), dim_ptr());
}

void DenseShapeMutView::set_shape(const DenseShapeView& shape) {
  CHECK_EQ(NumAxes(), shape.NumAxes());
  std::copy(shape.ptr(), shape.ptr() + shape.NumAxes(), dim_ptr());
}

}  // namespace oneflow
