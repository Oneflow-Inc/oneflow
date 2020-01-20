#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

Tensor::Tensor(const Tensor& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  mut_shape_.reset(new MutShapeView(*rhs.mut_shape_));
  data_type_ = rhs.data_type_;
}

Tensor::Tensor(Blob* blob) {
  dptr_ = blob->mut_dptr();
  shape_ = blob->shape();
  mut_shape_.reset(new MutShapeView(*blob->mut_shape_view()));
  data_type_ = blob->data_type();
}

}  // namespace user_op

}  // namespace oneflow
