#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

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
  dim0_valid_num_ptr_ = header_pod_ptr_.MutTensorPtr<int64_t>(FieldKey::kDim0ValidNum, nullptr);
  dim1_valid_num_ptr_ = header_pod_ptr_.MutTensorPtr<int64_t>(FieldKey::kDim1ValidNum, nullptr);
  dim2_valid_num_ptr_ = header_pod_ptr_.MutTensorPtr<int64_t>(FieldKey::kDim2ValidNum, nullptr);
  record_id_in_device_piece_ptr_ =
      header_pod_ptr_.MutTensorPtr<int64_t>(FieldKey::kRecordIdInDevicePiece, nullptr);
  dptr_ = body_ptr;
  dynamic_shape_ = blob_desc->shape();
  record_num_ = -1;
}

const char* Blob::data_id(int32_t no) const { UNIMPLEMENTED(); }

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

int64_t Blob::dim0_valid_num(int64_t no) const {
  CHECK_GE(no, 0);
  CHECK_LT(no, dim0_inner_shape().At(0));
  int64_t val;
  if (dim0_valid_num_ptr_) {
    val = dim0_valid_num_ptr_[no];
    CHECK_GE(val, 0);
    CHECK_LE(val, dim0_inner_shape().Count(1));
  } else {
    val = dim0_inner_shape().Count(1);
  }
  return val;
}
void Blob::set_dim0_valid_num(int64_t no, int64_t val) {
  CHECK_NOTNULL(dim0_valid_num_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, dim0_inner_shape().At(0));
  CHECK_GE(val, 0);
  CHECK_LE(val, dim0_inner_shape().Count(1));
  dim0_valid_num_ptr_[no] = val;
}

int64_t Blob::dim1_valid_num(int64_t no) const {
  CHECK_GE(no, 0);
  CHECK_LT(no, blob_desc_->shape().At(0));
  int64_t val;
  if (dim1_valid_num_ptr_) {
    val = dim1_valid_num_ptr_[no];
    CHECK_GE(val, 0);
    CHECK_LE(val, blob_desc_->shape().At(1));
  } else {
    val = blob_desc_->shape().At(1);
  }
  return val;
}
void Blob::set_dim1_valid_num(int64_t no, int64_t val) {
  CHECK_NOTNULL(dim1_valid_num_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, blob_desc_->shape().At(0));
  CHECK_GE(val, 0);
  CHECK_LE(val, blob_desc_->shape().At(1));
  dim1_valid_num_ptr_[no] = val;
}

int64_t Blob::dim2_valid_num(int64_t dim0_idx, int64_t dim1_idx) const {
  CHECK_GE(dim0_idx, 0);
  CHECK_LT(dim0_idx, blob_desc_->shape().At(0));
  CHECK_GE(dim1_idx, 0);
  CHECK_LT(dim1_idx, blob_desc_->shape().At(1));
  int64_t val;
  if (dim2_valid_num_ptr_) {
    val = *(dim2_valid_num_ptr_ + dim0_idx * blob_desc_->shape().At(1) + dim1_idx);
    CHECK_GE(val, 0);
    CHECK_LE(val, blob_desc_->shape().At(2));
  } else {
    val = blob_desc_->shape().At(2);
  }
  return val;
}

int64_t Blob::record_id_in_device_piece(int64_t no) const {
  CHECK_GE(no, 0);
  CHECK_LT(no, shape().At(0));
  int64_t val;
  if (record_id_in_device_piece_ptr_) {
    val = record_id_in_device_piece_ptr_[no];
    CHECK_GE(val, 0);
  } else {
    val = no;
  }
  return val;
}

void Blob::set_record_id_in_device_piece(int64_t no, int64_t val) {
  CHECK_NOTNULL(record_id_in_device_piece_ptr_);
  CHECK_GE(no, 0);
  CHECK_LT(no, static_shape().At(0));
  CHECK_GE(val, 0);
  record_id_in_device_piece_ptr_[no] = val;
}

void Blob::set_dim2_valid_num(int64_t dim0_idx, int64_t dim1_idx, int64_t val) {
  CHECK_NOTNULL(dim2_valid_num_ptr_);
  CHECK_GE(dim0_idx, 0);
  CHECK_LT(dim0_idx, blob_desc_->shape().At(0));
  CHECK_GE(dim1_idx, 0);
  CHECK_LT(dim1_idx, blob_desc_->shape().At(1));
  CHECK_GE(val, 0);
  CHECK_LE(val, blob_desc_->shape().At(2));
  *(dim2_valid_num_ptr_ + dim0_idx * blob_desc_->shape().At(1) + dim1_idx) = val;
}

const Shape& Blob::shape() const {
  if (dim0_valid_num_ptr_ == nullptr) { return static_shape(); }
  return dynamic_shape();
}

size_t Blob::ContiguousDim0ValidNum() const {
  size_t contiguous_invalid_instance_num = 0;
  for (int i = dim0_inner_shape().At(0) - 1; i >= 0; --i) {
    size_t valid_num = dim0_valid_num(i);
    contiguous_invalid_instance_num += dim0_inner_shape().Count(1) - valid_num;
    if (valid_num > 0) { break; }
  }
  return blob_desc_->shape().At(0) - contiguous_invalid_instance_num;
}

bool Blob::IsShapeEmpty() const {
  if (dim0_valid_num_ptr_ == nullptr) { return false; }
  return ContiguousDim0ValidNum() == 0;
}

const Shape& Blob::dynamic_shape() const {
  size_t contiguous_instance_num = ContiguousDim0ValidNum();
  CHECK_LE(contiguous_instance_num, static_shape().At(0));
  if (dynamic_shape_.At(0) != contiguous_instance_num) {
    dynamic_shape_.Set(0, contiguous_instance_num);
  }
  return dynamic_shape_;
}

const int32_t& Blob::record_num() const { return record_num_; }
void Blob::set_record_num(int32_t val) { record_num_ = val; }
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

void Blob::CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  CHECK_EQ(rhs->shape().elem_cnt(), shape().elem_cnt());
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfValidDataContent(), mem_case(),
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

size_t Blob::ByteSizeOfDim0ValidNumField() const {
  return blob_desc_->ByteSizeOfDim0ValidNumField();
}

size_t Blob::ByteSizeOfDim1ValidNumField() const {
  return blob_desc_->ByteSizeOfDim1ValidNumField();
}

size_t Blob::ByteSizeOfDim2ValidNumField() const {
  return blob_desc_->ByteSizeOfDim2ValidNumField();
}

size_t Blob::ByteSizeOfRecordIdInDevicePieceField() const {
  return blob_desc_->ByteSizeOfRecordIdInDevicePieceField();
}

size_t Blob::ByteSizeOfValidDataContent() const {
  return shape().elem_cnt() * GetSizeOfDataType(data_type());
}

void Blob::CopyDim0ValidNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfDim0ValidNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfDim0ValidNumField(), rhs->ByteSizeOfDim0ValidNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_dim0_valid_num_ptr(), rhs->dim0_valid_num_ptr(),
                           ByteSizeOfDim0ValidNumField());
}

void Blob::CopyDim1ValidNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfDim1ValidNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfDim1ValidNumField(), rhs->ByteSizeOfDim1ValidNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_dim1_valid_num_ptr(), rhs->dim1_valid_num_ptr(),
                           ByteSizeOfDim1ValidNumField());
}

void Blob::CopyDim2ValidNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfDim2ValidNumField() == 0) { return; }
  CHECK_EQ(ByteSizeOfDim2ValidNumField(), rhs->ByteSizeOfDim2ValidNumField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_dim2_valid_num_ptr(), rhs->dim2_valid_num_ptr(),
                           ByteSizeOfDim2ValidNumField());
}

void Blob::CopyRecordIdInDevicePieceFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || ByteSizeOfRecordIdInDevicePieceField() == 0) { return; }
  CHECK_EQ(ByteSizeOfRecordIdInDevicePieceField(), rhs->ByteSizeOfRecordIdInDevicePieceField());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_record_id_in_device_piece_ptr(),
                           rhs->record_id_in_device_piece_ptr(),
                           ByteSizeOfRecordIdInDevicePieceField());
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

size_t Blob::CalcDim0ValidNumSum() const {
  if (dim0_valid_num_ptr() == nullptr) { return static_shape().At(0); }
  size_t sum = 0;
  FOR_RANGE(int64_t, i, 0, dim0_inner_shape().At(0)) { sum += dim0_valid_num(i); }
  return sum;
}

}  // namespace oneflow
