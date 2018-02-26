#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/register/blob_implement.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define MAKE_BLOB_ENTRY(data_type_pair, ndims, device_type)                   \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair), ndims, device_type), [=]() { \
     return new BlobImpl<OF_PP_PAIR_FIRST(data_type_pair), ndims,             \
                         device_type>(regst, blob_desc, mem_ptr,              \
                                      comm_net_token);                        \
   }},

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
              const void* comm_net_token, DeviceType device_type) {
  static const HashMap<std::string, std::function<Blob*()>> creators = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_BLOB_ENTRY, ALL_DATA_TYPE_SEQ,
                                       DIM_SEQ, DEVICE_TYPE_SEQ)};
  return creators.at(GetHashKey(
      blob_desc->data_type(),
      static_cast<int32_t>(blob_desc->shape().NumAxes()), device_type))();
}

Blob::Blob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token) {
  mem_ptr_ = mem_ptr;
  if (blob_desc->has_data_id_field()) {
    data_id_ptr_ = mem_ptr;
  } else {
    data_id_ptr_ = nullptr;
  }
  if (blob_desc->has_col_num_field()) {
    col_num_ptr_ = reinterpret_cast<int32_t*>(
        mem_ptr + blob_desc->ByteSizeOfDataIdField());
  } else {
    col_num_ptr_ = nullptr;
  }
  dptr_ = mem_ptr + blob_desc->ByteSizeOfDataIdField()
          + blob_desc->ByteSizeOfColNumField();
  blob_desc_ = blob_desc;
  comm_net_token_ = comm_net_token;
  regst_ = regst;
}

const char* Blob::data_id(int32_t no) const {
  CHECK_NOTNULL(data_id_ptr_);
  return data_id_ptr_ + no * JobDesc::Singleton()->SizeOfOneDataId();
}

int32_t Blob::col_num(int32_t no) const {
  CHECK_NOTNULL(col_num_ptr_);
  return *(col_num_ptr_ + no);
}

void Blob::set_col_num(int32_t no, int32_t val) {
  CHECK_NOTNULL(col_num_ptr_);
  *(col_num_ptr_ + no) = val;
}

size_t Blob::ByteSizeOfDataIdField() const {
  return blob_desc_->ByteSizeOfDataIdField();
}

size_t Blob::ByteSizeOfColNumField() const {
  return blob_desc_->ByteSizeOfColNumField();
}

size_t Blob::ByteSizeOfDataContentField() const {
  return blob_desc_->ByteSizeOfDataContentField();
}

template<DeviceType device_type>
void Blob::CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, dptr_, rhs->dptr_,
                      ByteSizeOfDataContentField());
}

template<DeviceType device_type>
void Blob::CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, data_id_ptr_, rhs->data_id_ptr_,
                      ByteSizeOfDataIdField());
}

template<DeviceType device_type>
void Blob::CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, col_num_ptr_, rhs->col_num_ptr_,
                      ByteSizeOfColNumField());
}

template<DeviceType device_type>
void Blob::CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, mut_memory_ptr(), rhs->memory_ptr(),
                      TotalByteSize());
}

int32_t Blob::col_id() const { return regst_->col_id(); }

void Blob::set_col_id(int32_t val) { regst_->set_col_id(val); }

int32_t Blob::max_col_id() const { return regst_->max_col_id(); }

void Blob::set_max_col_id(int32_t val) { regst_->set_max_col_id(val); }

bool Blob::IsColValid() const { return col_id() <= max_col_id(); }

#define INSTANTIATE_BLOB_FUNC(dev_t)                                       \
  template void Blob::CopyDataContentFrom<dev_t>(DeviceCtx*, const Blob*); \
  template void Blob::CopyDataIdFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyColNumFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyFrom<dev_t>(DeviceCtx*, const Blob*);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BLOB_FUNC, DEVICE_TYPE_SEQ);

}  // namespace oneflow
