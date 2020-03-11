#include "oneflow/core/framework/tensor_desc.h"

namespace oneflow {

namespace user_op {

TensorDesc::TensorDesc(const TensorDesc& rhs) { *this = rhs; }

TensorDesc& TensorDesc::operator=(const TensorDesc& rhs) {
  shape_ = rhs.shape_;
  data_type_ = rhs.data_type_;
  is_dynamic_ = rhs.is_dynamic_;
  is_tensor_list_ = rhs.is_tensor_list_;
  return *this;
}

TensorDesc::TensorDesc(const BlobDescProto& proto) { *this = proto; }

TensorDesc& TensorDesc::operator=(const BlobDescProto& proto) {
  CHECK(proto.header_is_opaque() == false);
  data_type_ = proto.body().data_type();
  shape_ = Shape(proto.body().shape());
  is_dynamic_ = proto.is_dynamic();
  is_tensor_list_ = proto.is_tensor_list();
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
