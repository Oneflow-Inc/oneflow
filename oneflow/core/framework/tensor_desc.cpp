#include "oneflow/core/framework/tensor_desc.h"

namespace oneflow {

namespace user_op {

TensorDesc::TensorDesc(const TensorDesc& rhs) { *this = rhs; }

TensorDesc& TensorDesc::operator=(const TensorDesc& rhs) {
  shape_ = rhs.shape_;
  data_type_ = rhs.data_type_;
  return *this;
}

TensorDesc::TensorDesc(const BlobDescProto& proto) { *this = proto; }

TensorDesc& TensorDesc::operator=(const BlobDescProto& proto) {
  data_type_ = proto.body().data_type();
  shape_ = Shape(proto.body().shape());
  return *this;
}
}  // namespace user_op

}  // namespace oneflow
