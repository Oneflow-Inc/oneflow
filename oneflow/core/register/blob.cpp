#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/register/blob_implement.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

Blob::Blob(Regst* regst, const BlobDesc* blob_desc, char* hptr, char* dptr)
    : hptr_(hptr),
      dptr_(dptr),
      data_id_ptr_(nullptr),
      col_num_ptr_(nullptr),
      blob_desc_(blob_desc),
      regst_(regst) {
  CHECK(hptr || dptr);
  if (hptr) {
    char* offset = hptr;
    if (has_data_id_field()) {
      data_id_ptr_ = offset;
      // offset += RoundUp(ByteSizeOfDataIdField(), kCudaAlignSize);
      offset += ByteSizeOfDataIdField();
    }
    if (blob_desc->has_col_num_field()) {
      col_num_ptr_ = reinterpret_cast<int32_t*>(offset);
      // offset += RoundUp(ByteSizeOfColNumField(), kCudaAlignSize);
      offset += ByteSizeOfColNumField();
    }
    if (!dptr_) { dptr_ = offset; }
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

int32_t Blob::col_id() const {
  CHECK_NOTNULL(regst_);
  return regst_->col_id();
}
void Blob::set_col_id(int32_t val) {
  CHECK_NOTNULL(regst_);
  regst_->set_col_id(val);
}
int32_t Blob::max_col_id() const {
  CHECK_NOTNULL(regst_);
  return regst_->max_col_id();
}
void Blob::set_max_col_id(int32_t val) {
  CHECK_NOTNULL(regst_);
  regst_->set_max_col_id(val);
}
const MemoryCase& Blob::mem_case() const {
  CHECK_NOTNULL(regst_);
  return regst_->regst_desc()->mem_case();
}

bool Blob::IsMemoryContinuous() const {
  return mem_case().has_host_mem() && !mem_case().host_mem().used_by_device();
}

void Blob::InitOFRecordBlobIfNeed() {
  if (blob_desc_->data_type() == kOFRecord) {
    int64_t elem_cnt = blob_desc_->shape().elem_cnt();
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      Global<MemoryAllocator>::Get()->PlacementNew(&(mut_dptr<OFRecord>()[i]));
    }
  }
}

#define MAKE_BLOB_ENTRY(data_type_pair, ndims, device_type)                                      \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair), ndims, device_type),                            \
   [](Regst* regst, const BlobDesc* blob_desc, char* hptr, char* dptr) {                         \
     return new BlobImpl<OF_PP_PAIR_FIRST(data_type_pair), ndims, device_type>(regst, blob_desc, \
                                                                               hptr, dptr);      \
   }},

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* hptr, char* dptr,
              DeviceType device_type) {
  static const HashMap<std::string, std::function<Blob*(Regst*, const BlobDesc*, char*, char*)>>
      creators = {OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_BLOB_ENTRY, ALL_DATA_TYPE_SEQ, DIM_SEQ,
                                                   DEVICE_TYPE_SEQ)};
  std::string key = GetHashKey(blob_desc->data_type(),
                               static_cast<int32_t>(blob_desc->shape().NumAxes()), device_type);
  return creators.at(key)(regst, blob_desc, hptr, dptr);
}

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr, DeviceType device_type) {
  return NewBlob(regst, blob_desc, nullptr, mem_ptr, device_type);
}

}  // namespace oneflow
