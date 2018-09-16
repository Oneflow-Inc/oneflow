#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr) {
  Init(regst, blob_desc, header_ptr, header_ptr + blob_desc->ByteSizeOfBlobHeader());
}

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr) {
  Init(regst, blob_desc, header_ptr, body_ptr);
}

void Blob::Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr) {
  if (body_ptr == header_ptr + blob_desc->ByteSizeOfBlobHeader()) {
    is_contiguous_ = true;
  } else {
    is_contiguous_ = false;
  }

  regst_ = regst;
  blob_desc_ = blob_desc;
  header_ptr_ = header_ptr;
  if (blob_desc->has_data_id_field()) {
    data_id_ptr_ = header_ptr;
  } else {
    data_id_ptr_ = nullptr;
  }
  char* offset = header_ptr + blob_desc->ByteSizeOfDataIdField();
  if (blob_desc->has_col_num_field()) {
    col_num_ptr_ = reinterpret_cast<int32_t*>(offset);
  } else {
    col_num_ptr_ = nullptr;
  }
  offset = header_ptr + blob_desc->ByteSizeOfDataIdField() + blob_desc->ByteSizeOfColNumField();
  if (blob_desc->has_instance_num_field()) {
    instance_num_ptr_ = reinterpret_cast<int32_t*>(offset);
  } else {
    instance_num_ptr_ = nullptr;
  }
  dptr_ = body_ptr;
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

void Blob::CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfBlobHeader() == 0) { return; }
  CHECK_EQ(ByteSizeOfBlobHeader(), rhs->ByteSizeOfBlobHeader());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_header_ptr(), rhs->header_ptr(), ByteSizeOfBlobHeader());
}

void Blob::CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfDataIdField() == 0) { return; }
  CHECK_EQ(ByteSizeOfDataIdField(), rhs->ByteSizeOfDataIdField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_data_id(), rhs->data_id(), ByteSizeOfDataIdField());
}

void Blob::CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfColNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfColNumField(), rhs->ByteSizeOfColNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_col_num(), rhs->col_num(), ByteSizeOfColNumField());
}

void Blob::CopyInstanceNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfInstanceNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfInstanceNumField(), rhs->ByteSizeOfInstanceNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_instance_num(), rhs->instance_num(),
                           ByteSizeOfInstanceNumField());
}

void Blob::AccumulateInstanceNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfInstanceNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfInstanceNumField(), rhs->ByteSizeOfInstanceNumField());
  KernelUtil<DeviceType::kCPU, int32_t>::Axpy(device_ctx, 1, 1, rhs->instance_num(), 1,
                                              mut_instance_num(), 1);
}

void Blob::AccumulateInstanceNumInPackedHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  const int32_t* instance_num_from_ptr = reinterpret_cast<const int32_t*>(
      static_cast<const char*>(rhs->header_ptr()) + rhs->ByteSizeOfDataIdField()
      + rhs->ByteSizeOfColNumField());
  int32_t* instance_num_to_ptr =
      reinterpret_cast<int32_t*>(static_cast<char*>(this->mut_header_ptr())
                                 + rhs->ByteSizeOfDataIdField() + rhs->ByteSizeOfColNumField());
  KernelUtil<DeviceType::kCPU, int32_t>::Axpy(device_ctx, 1, 1, instance_num_from_ptr, 1,
                                              instance_num_to_ptr, 1);
}

void Blob::CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  if (is_contiguous_) {
    CHECK_EQ(TotalByteSize(), rhs->TotalByteSize());
    AutoMemcpy(device_ctx, mut_header_ptr(), rhs->header_ptr(), TotalByteSize(), mem_case(),
               rhs->mem_case());
  } else {
    CopyHeaderFrom(device_ctx, rhs);
    CopyDataContentFrom(device_ctx, rhs);
  }
}

}  // namespace oneflow
