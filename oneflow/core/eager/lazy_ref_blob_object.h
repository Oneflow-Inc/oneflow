#ifndef ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/eager/blob_object.h"

namespace oneflow {
namespace eager {

class LazyRefBlobObject : public BlobObject {
 public:
  LazyRefBlobObject(const LazyRefBlobObject&) = delete;
  LazyRefBlobObject(LazyRefBlobObject&&) = delete;
  LazyRefBlobObject(Blob* blob)
      : BlobObject(std::make_shared<MemoryCase>(blob->mem_case()), blob->data_type()) {
    rt_blob_desc_.reset(new RtBlobDesc(blob_desc()));
    ref_blob_ = blob;
  }
  virtual ~LazyRefBlobObject() override = default;

  virtual const Blob& blob() const override { return *ref_blob_; }
  virtual Blob* mut_blob() override { return ref_blob_; }

  // is it legal?
  // BlobDesc* mut_blob_desc() { return &blob_desc_; }

  virtual Maybe<void> TryInitBlob() override { return Error::Unimplemented(); };

  // TODO(daquexian):
  // virtual void TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;

 private:
  Blob* ref_blob_ = nullptr;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_
