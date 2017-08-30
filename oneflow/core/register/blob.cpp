#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

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

void Blob::set_data_id(int32_t no, const std::string& data_id) {
  size_t max_length = JobDesc::Singleton()->SizeOfOneDataId();
  char* ptr = new char[max_length];
  for (size_t i = 0; i < data_id.length() && i < max_length; ++i) {
    *(ptr + i) = data_id[i];
  }
  for (size_t i = data_id.length(); i < max_length; ++i) { *(ptr + i) = '\0'; }
  CudaCheck(cudaMemcpy(mut_data_id(no), ptr, max_length, cudaMemcpyHostToHost));
  delete ptr;
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
