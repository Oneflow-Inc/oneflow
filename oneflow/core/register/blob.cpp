#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

Blob::Blob(const BlobDesc* blob_desc, char* mem_ptr) {
  data_id_ptr_ = blob_desc->has_data_id() ? mem_ptr : nullptr;
  dptr_ = mem_ptr + blob_desc->ByteSizeOfDataIdField();
  blob_desc_ = blob_desc;
}

const char* Blob::data_id(int32_t no) const {
  CHECK_NOTNULL(data_id_ptr_);
  return data_id_ptr_ + no * JobDesc::Singleton()->SizeOfOneDataId();
}

}  // namespace oneflow
