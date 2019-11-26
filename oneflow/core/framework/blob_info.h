#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_INFO_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_INFO_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace user_op {

class BlobInfo final {
 public:
  BlobInfo() = default;
  ~BlobInfo() = default;
  BlobInfo(const BlobInfo&);
  BlobInfo(const BlobDescProto&);
  BlobInfo(const Shape& shape, DataType dtype) : shape_(shape), data_type_(dtype) {}

  BlobInfo& operator=(const BlobInfo&);
  BlobInfo& operator=(const BlobDescProto&);

  const Shape& shape() const { return shape_; }
  const DataType& data_type() const { return data_type_; }

 private:
  Shape shape_;
  DataType data_type_;
};

}  // namespace user_op

}  // namespace oneflow

#endif
