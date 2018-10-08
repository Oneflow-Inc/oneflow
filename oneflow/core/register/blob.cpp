#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr)
    : header_pod_ptr_(blob_desc->header_pod_desc(), header_ptr) {
  Init(regst, blob_desc, header_ptr, header_ptr + blob_desc->ByteSizeOfBlobHeader());
}

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr)
    : header_pod_ptr_(blob_desc->header_pod_desc(), header_ptr) {
  Init(regst, blob_desc, header_ptr, body_ptr);
}

void Blob::Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr) {
  is_contiguous_ = (body_ptr == header_ptr + blob_desc->ByteSizeOfBlobHeader());
  regst_ = regst;
  blob_desc_ = blob_desc;
  header_ptr_ = header_ptr;
  data_id_ptr_ = header_pod_ptr_.MutTensorPtr<char>(FieldKey::kDataId, nullptr);
  col_num_ptr_ = header_pod_ptr_.MutTensorPtr<int32_t>(FieldKey::kColNum, nullptr);
  varying_instance_num_ptr_ =
      header_pod_ptr_.MutTensorPtr<int32_t>(FieldKey::kVaryingInstanceNum, nullptr);
  instance_varying_elem_cnt_ptr_ =
      header_pod_ptr_.MutTensorPtr<int32_t>(FieldKey::kInstanceVaryingElemCnt, nullptr);
  dptr_ = body_ptr;
  dynamic_shape_ = blob_desc->shape();
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

int32_t Blob::instance_varying_elem_cnt(int32_t no) const {
  CHECK_NOTNULL(instance_varying_elem_cnt_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, blob_desc_->shape().At(0));
  return instance_varying_elem_cnt_ptr_[no];
}

void Blob::set_instance_varying_elem_cnt(int32_t no, int32_t val) {
  CHECK_NOTNULL(instance_varying_elem_cnt_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, blob_desc_->shape().At(0));
  CHECK_GE(val, 0);
  CHECK_LT(val, blob_desc_->shape().Count(1));
  instance_varying_elem_cnt_ptr_[no] = val;
}

int32_t Blob::varying_instance_num(int32_t no) const {
  CHECK_NOTNULL(varying_instance_num_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, instance_inner_shape()->At(0));
  return varying_instance_num_ptr_[no];
}

void Blob::set_varying_instance_num(int32_t no, int32_t val) {
  CHECK_NOTNULL(varying_instance_num_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, instance_inner_shape()->At(0));
  CHECK_GE(val, 0);
  CHECK_LT(val, instance_inner_shape()->Count(1));
  varying_instance_num_ptr_[no] = val;
}


int32_t Blob::instance_available_elem_cnt(int32_t no) const {
  if (instance_varying_elem_cnt_ptr_ != nullptr) { return instance_varying_elem_cnt(no); }
  return blob_desc_->shape().Count(1);
}

int32_t Blob::instance_available_elem_cnt() const {
  if (instance_varying_elem_cnt_ptr_ != nullptr) { 
    size_t num = 0;
    FOR_RANGE(int, i, 0, blob_desc_->shape().At(0)){ num += instance_varying_elem_cnt(i); }
    return num;
  }
  return blob_desc_->shape().Count(1);
}

int32_t Blob::available_instance_num(int32_t no) const {
  if (varying_instance_num_ptr_ != nullptr) { return varying_instance_num(no); }
  return instance_inner_shape()->Count(1);
}

int32_t Blob::available_instance_num() const {
  if (varying_instance_num_ptr_ != nullptr) {
    size_t num = 0;
    FOR_RANGE(int, i, 0, instance_inner_shape()->At(0)) { num += varying_instance_num(i); }
    return num;
  }
  return blob_desc_->shape().At(0);
}

const Shape& Blob::shape() const {
  if (varying_instance_num_ptr_ == nullptr) { return static_shape(); }
  return dynamic_shape();
}

const Shape& Blob::dynamic_shape() const {
  size_t last_invalid_instance_num =
      instance_inner_shape()->Count(1) - varying_instance_num(instance_inner_shape()->At(0) - 1);
  size_t contiguous_instance_num = blob_desc_->shape().At(0) - last_invalid_instance_num;
  if (dynamic_shape_.At(0) != contiguous_instance_num) {
    dynamic_shape_.Set(0, contiguous_instance_num);
  }
  return dynamic_shape_;
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

size_t Blob::ByteSizeOfVaryingInstanceNumField() const {
  return blob_desc_->ByteSizeOfVaryingInstanceNumField();
}

size_t Blob::ByteSizeOfInstanceVaryingElemCntField() const {
  return blob_desc_->ByteSizeOfInstanceVaryingElemCntField();
}

void Blob::CopyVaryingInstanceNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfVaryingInstanceNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfVaryingInstanceNumField(), rhs->ByteSizeOfVaryingInstanceNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_varying_instance_num(), rhs->varying_instance_num(),
                           ByteSizeOfVaryingInstanceNumField());
}

void Blob::CopyInstanceVaryingElemCntFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfVaryingInstanceNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfInstanceVaryingElemCntField(), rhs->ByteSizeOfInstanceVaryingElemCntField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_instance_varying_elem_cnt(),
                           rhs->instance_varying_elem_cnt(),
                           ByteSizeOfInstanceVaryingElemCntField());
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
