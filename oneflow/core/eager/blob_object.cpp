#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace eager {

Maybe<void> BlobObject::TryInitBlob() {
  if (!blob_) { JUST(InitBlob()); }
  return Maybe<void>::Ok();
}

Maybe<void> BlobObject::InitBlob() {
  CHECK_NE_OR_RETURN(blob_desc_.data_type(), DataType::kInvalidDataType);
  rt_blob_desc_.reset(new RtBlobDesc(blob_desc_));
  {
    header_buffer_.reset();
    int64_t header_byte_size = rt_blob_desc_->ByteSizeOfBlobHeader();
    const auto& FreeHeader = [header_byte_size](char* dptr) { std::free(dptr); };
    char* ptr = reinterpret_cast<char*>(std::malloc(header_byte_size));
    header_buffer_ = std::unique_ptr<char, std::function<void(char*)>>(ptr, FreeHeader);
  }
  blob_.reset(new Blob(*mem_case_, rt_blob_desc_.get(), header_buffer_.get(), nullptr));
  return Maybe<void>::Ok();
}

void BlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL(allocator);
  Blob* blob = mut_blob();
  CHECK_NOTNULL(blob);
  const std::size_t required_body_bytes = blob->AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL(blob->dptr());
    return;
  }
  if (blob->dptr() != nullptr) {
    CHECK_EQ(blob_body_bytes_, required_body_bytes);
    return;
  }
  {
    // reset blob_dptr_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
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
