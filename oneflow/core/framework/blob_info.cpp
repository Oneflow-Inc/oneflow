#include "oneflow/core/framework/blob_info.h"

namespace oneflow {

namespace user_op {

BlobInfo::BlobInfo(const BlobInfo& rhs) { *this = rhs; }

BlobInfo& BlobInfo::operator=(const BlobInfo& rhs) {
  shape_ = rhs.shape_;
  data_type_ = rhs.data_type_;
  return *this;
}

BlobInfo::BlobInfo(const BlobDescProto& proto) { *this = proto; }

BlobInfo& BlobInfo::operator=(const BlobDescProto& proto) {
  data_type_ = proto.body().data_type();
  shape_ = Shape(proto.body().shape());
  return *this;
}
}  // namespace user_op

}  // namespace oneflow
