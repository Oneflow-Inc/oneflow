#ifndef ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {
namespace eager {

class BlobObjectType : public vm::Object {
 public:
  BlobObjectType(const BlobObjectType&) = delete;
  BlobObjectType(BlobObjectType&&) = delete;
  BlobObjectType() : blob_desc_(DataType::kInvalidDataType) {}
  ~BlobObjectType() = default;

  const BlobDesc& blob_desc() const { return blob_desc_; }
  BlobDesc* mut_blob_desc() { return &blob_desc_; }

 private:
  BlobDesc blob_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
