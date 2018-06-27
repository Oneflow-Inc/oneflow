#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/register/blob_implement.h"
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
  if (blob_desc_->data_type() == kOFRecordPtr) {
    int64_t elem_cnt = blob_desc->shape().elem_cnt();
    OFRecordPtr* ofrecord_ptr = static_cast<OFRecordPtr*>(dptr_);
    FOR_RANGE(int64_t, i, 0, elem_cnt) { *(ofrecord_ptr + i) = new OFRecord(); }
  }
}

Blob::~Blob() {
  if (blob_desc_->data_type() == kOFRecordPtr) {
    int64_t elem_cnt = blob_desc_->shape().elem_cnt();
    OFRecordPtr* ofrecord_ptr = static_cast<OFRecordPtr*>(dptr_);
    FOR_RANGE(int64_t, i, 0, elem_cnt) { delete *(ofrecord_ptr + i); }
  }
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

#define MAKE_BLOB_ENTRY(data_type_pair, ndims, device_type)                                      \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair), ndims, device_type),                            \
   [](Regst* regst, const BlobDesc* blob_desc, char* mem_ptr) {                                  \
     return new BlobImpl<OF_PP_PAIR_FIRST(data_type_pair), ndims, device_type>(regst, blob_desc, \
                                                                               mem_ptr);         \
   }},

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr, DeviceType device_type) {
  static const HashMap<
      std::string, std::function<Blob*(Regst * regst, const BlobDesc* blob_desc, char* mem_ptr)>>
      creators = {OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_BLOB_ENTRY, ALL_POD_DATA_TYPE_SEQ, DIM_SEQ,
                                                   DEVICE_TYPE_SEQ)};
  std::string key = GetHashKey(blob_desc->data_type(),
                               static_cast<int32_t>(blob_desc->shape().NumAxes()), device_type);
  return creators.at(key)(regst, blob_desc, mem_ptr);
}

}  // namespace oneflow
