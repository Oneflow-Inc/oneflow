#ifndef ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace eager {

class BlobObject : public vm::Object {
 public:
  BlobObject(const BlobObject&) = delete;
  BlobObject(BlobObject&&) = delete;
  BlobObject(const std::shared_ptr<MemoryCase>& mem_case, DataType data_type)
      : mem_case_(mem_case), blob_body_bytes_(0), blob_desc_(data_type) {}
  virtual ~BlobObject() override = default;

  const BlobDesc& blob_desc() const { return blob_desc_; }
  BlobDesc* mut_blob_desc() { return &blob_desc_; }

  virtual const Blob& blob() const { return *blob_; }
  virtual Blob* mut_blob() { return blob_.get(); }
  virtual Maybe<void> TryInitBlob();

  void TryAllocateBlobBodyMemory(DeviceCtx* device_ctx);

 private:
  Maybe<void> InitBlob();

  std::shared_ptr<MemoryCase> mem_case_;
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char, std::function<void(char*)>> header_buffer_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::size_t blob_body_bytes_;

 protected:
  BlobDesc blob_desc_;
  std::unique_ptr<RtBlobDesc> rt_blob_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
