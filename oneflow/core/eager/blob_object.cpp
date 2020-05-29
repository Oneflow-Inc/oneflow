#include "oneflow/core/eager/blob_object.h"

namespace oneflow {
namespace eager {

Blob* BlobObject::mutable_blob() {
  if (!blob_) { InitBlob(); }
  return blob_.get();
}

void BlobObject::InitBlob() {
  rt_blob_desc_.reset(new RtBlobDesc(blob_desc_));
  header_buffer_.reset(new char[rt_blob_desc_->AlignedByteSizeOfBlobBody()]);
  blob_.reset(new Blob(*mem_case_, rt_blob_desc_.get(), header_buffer_.get(), nullptr));
}

void BlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL(allocator);
  Blob* blob = mut_blob();
  CHECK_NOTNULL(blob);
  std::size_t required_body_bytes = blob->AlignedByteSizeOfBlobBody();
  if (blob->dptr() != nullptr && blob_body_bytes_ == required_body_bytes) { return; }
  {
    // reset blob_dptr_;
    auto Free = [allocator, required_body_bytes](char* dptr) {
      allocator->Deallocate(dptr, required_body_bytes);
    };
    char* dptr = nullptr;
    blob_dptr_.reset();
    allocator->Allocate(&dptr, required_body_bytes);
    blob_dptr_ = std::unique_ptr<char, std::function<void(char*)>>(dptr, Free);
    blob->reset_dptr(dptr);
  }
  blob_body_bytes_ = required_body_bytes;
}

}  // namespace eager
}  // namespace oneflow
