#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class BlobDesc final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  BlobDesc() = default;
  ~BlobDesc() = default;

  BlobDesc(const BlobDescProto& proto) { shape_ = Shape(proto.shape()); }

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  void ToProto(BlobDescProto* proto) const {
    shape_.ToProto(proto->mutable_shape());
  }

 private:
  Shape shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
