#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

Tensor::Tensor(Blob* blob) {
  dptr_ = blob->mut_dptr();
  shape_ = blob->shape();
  if (blob->mut_shape_view()) {
    mut_shape_.reset(new MutShapeView(*blob->mut_shape_view()));
  } else {
    mut_shape_.reset();
  }
  data_type_ = blob->data_type();
  mem_case_ = &(blob->mem_case());
}

void Tensor::CopyWithoutData(const Tensor& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  if (rhs.mut_shape_) {
    mut_shape_.reset(new MutShapeView(*rhs.mut_shape_));
  } else {
    mut_shape_.reset();
  }
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  mut_shape_ = std::move(rhs.mut_shape_);
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
