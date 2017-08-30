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

template<>
void Blob::SetDataId<DeviceType::kCPU>(int32_t no, const std::string& data_id) {
  size_t max_length = JobDesc::Singleton()->SizeOfOneDataId();
  CHECK_GE(max_length, data_id.length());
  memcpy(mut_data_id(no), data_id.c_str(), data_id.length());
  memset(mut_data_id(no) + data_id.length(), '\0',
         max_length - data_id.length());
}

template<>
void Blob::SetDataId<DeviceType::kGPU>(int32_t no, const std::string& data_id) {
  size_t max_length = JobDesc::Singleton()->SizeOfOneDataId();
  CHECK_GE(max_length, data_id.length());
  CudaCheck(cudaMemcpy(mut_data_id(no), data_id.c_str(), data_id.length(),
                       cudaMemcpyHostToHost));
  CudaCheck(cudaMemset(mut_data_id(no) + data_id.length(), '\0',
                       max_length - data_id.length()));
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
