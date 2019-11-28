#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_DEF_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_DEF_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace user_op {

class BlobDef final {
 public:
  BlobDef() = default;
  ~BlobDef() = default;
  BlobDef(const BlobDef&);
  BlobDef(const BlobDescProto&);
  BlobDef(const Shape& shape, DataType dtype) : shape_(shape), data_type_(dtype) {}

  BlobDef& operator=(const BlobDef&);
  BlobDef& operator=(const BlobDescProto&);

  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }

 private:
  Shape shape_;
  DataType data_type_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BLOB_DEF_H_
