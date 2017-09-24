#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

Blob::Blob(const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token) {
  data_id_ptr_ = blob_desc->has_data_id() ? mem_ptr : nullptr;
  dptr_ = mem_ptr + blob_desc->ByteSizeOfDataIdField();
  blob_desc_ = blob_desc;
  comm_net_token_ = comm_net_token;
}

size_t Blob::data_id_len(int32_t no) const {
  if (*(data_id(no + 1) - 1) != '\0') {
    return JobDesc::Singleton()->SizeOfOneDataId();
  }
  return strlen(data_id_ptr_);
}

const char* Blob::data_id(int32_t no) const {
  CHECK_NOTNULL(data_id_ptr_);
  return data_id_ptr_ + no * JobDesc::Singleton()->SizeOfOneDataId();
}

template<>
void Blob::CopyDataIdFrom<DeviceType::kCPU>(DeviceCtx* device_ctx,
                                            const Blob* rhs) {
  Memcpy<DeviceType::kCPU>(device_ctx, data_id_ptr_, rhs->data_id_ptr_,
                           ByteSizeOfDataIdField(),
                           cudaMemcpyKind::cudaMemcpyHostToHost);
}

template<>
void Blob::CopyDataIdFrom<DeviceType::kGPU>(DeviceCtx* device_ctx,
                                            const Blob* rhs) {
  Memcpy<DeviceType::kGPU>(device_ctx, data_id_ptr_, rhs->data_id_ptr_,
                           ByteSizeOfDataIdField(),
                           cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

}  // namespace oneflow
