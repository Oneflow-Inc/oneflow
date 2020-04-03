#ifndef ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {
namespace eager {

class BlobObject : public vm::Object {
 public:
  BlobObject(const BlobObject&) = delete;
  BlobObject(BlobObject&&) = delete;
  BlobObject(const std::shared_ptr<MemoryCase>& mem_case, DataType data_type)
      : mem_case_(mem_case), blob_desc_(data_type) {}
  ~BlobObject() = default;

  const BlobDesc& blob_desc() const { return blob_desc_; }
  BlobDesc* mut_blob_desc() { return &blob_desc_; }

  Blob* mutable_blob();

 private:
  void InitBlob();

  std::shared_ptr<MemoryCase> mem_case_;
  BlobDesc blob_desc_;
  std::unique_ptr<RtBlobDesc> rt_blob_desc_;
  std::unique_ptr<char[]> header_buffer_;
  std::unique_ptr<Blob> blob_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
