#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace user_op {

Tensor::Tensor(const TensorDesc& def, char* dptr) : desc_(def), dptr_(dptr) {}

Tensor::Tensor(const Shape& shape, DataType dtype, char* dptr) : desc_(shape, dtype), dptr_(dptr) {}

Tensor::Tensor(const Tensor& rhs) {
  desc_ = rhs.desc_;
  dptr_ = rhs.dptr_;
}

}  // namespace user_op

}  // namespace oneflow
