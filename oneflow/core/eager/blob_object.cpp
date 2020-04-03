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

}  // namespace eager
}  // namespace oneflow
