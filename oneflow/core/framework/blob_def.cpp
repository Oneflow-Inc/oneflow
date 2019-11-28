#include "oneflow/core/framework/blob_def.h"

namespace oneflow {

namespace user_op {

BlobDef::BlobDef(const BlobDef& rhs) { *this = rhs; }

BlobDef& BlobDef::operator=(const BlobDef& rhs) {
  shape_ = rhs.shape_;
  data_type_ = rhs.data_type_;
  return *this;
}

BlobDef::BlobDef(const BlobDescProto& proto) { *this = proto; }

BlobDef& BlobDef::operator=(const BlobDescProto& proto) {
  data_type_ = proto.body().data_type();
  shape_ = Shape(proto.body().shape());
  return *this;
}
}  // namespace user_op

}  // namespace oneflow
