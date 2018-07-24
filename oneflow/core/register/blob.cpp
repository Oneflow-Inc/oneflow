#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Blob::Blob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr) {
  mem_ptr_ = mem_ptr;
  if (blob_desc->has_data_id_field()) {
    data_id_ptr_ = mem_ptr;
  } else {
    data_id_ptr_ = nullptr;
  }
  char* offset = mem_ptr + RoundUp(blob_desc->ByteSizeOfDataIdField(), kCudaAlignSize);
  if (blob_desc->has_col_num_field()) {
    col_num_ptr_ = reinterpret_cast<int32_t*>(offset);
  } else {
    col_num_ptr_ = nullptr;
  }
  dptr_ = offset + RoundUp(blob_desc->ByteSizeOfColNumField(), kCudaAlignSize);
  blob_desc_ = blob_desc;
  regst_ = regst;
}

const char* Blob::data_id(int32_t no) const {
  CHECK_NOTNULL(data_id_ptr_);
  return data_id_ptr_ + no * Global<JobDesc>::Get()->SizeOfOneDataId();
}

int32_t Blob::col_num(int32_t no) const {
  if (col_num_ptr_ == nullptr) {
    return 1;
  } else {
    return *(col_num_ptr_ + no);
  }
}

void Blob::set_col_num(int32_t no, int32_t val) {
  CHECK_NOTNULL(col_num_ptr_);
  *(col_num_ptr_ + no) = val;
}

int32_t Blob::col_id() const { return regst_->col_id(); }
void Blob::set_col_id(int32_t val) { regst_->set_col_id(val); }
int32_t Blob::max_col_id() const { return regst_->max_col_id(); }
void Blob::set_max_col_id(int32_t val) { regst_->set_max_col_id(val); }
const MemoryCase& Blob::mem_case() const { return regst_->regst_desc()->mem_case(); }

void Blob::CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfDataContentField(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_data_id(), rhs->data_id(), ByteSizeOfDataIdField(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_col_num(), rhs->col_num(), ByteSizeOfColNumField(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_memory_ptr(), rhs->memory_ptr(), TotalByteSize(), mem_case(),
             rhs->mem_case());
}

}  // namespace oneflow
