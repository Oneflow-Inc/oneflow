#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

Blob::Blob(const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token) {
  blob_header_ = reinterpret_cast<BlobHeader*>(mem_ptr);
  blob_header_->col_id = -1;
  blob_header_->max_col_id = -1;

  if (blob_desc->has_data_id_field()) {
    data_id_ptr_ = mem_ptr + blob_desc->ByteSizeOfBlobHeaderField();
  } else {
    data_id_ptr_ = nullptr;
  }
  if (blob_desc->has_col_num_field()) {
    col_num_ptr_ = reinterpret_cast<int32_t*>(
        mem_ptr + blob_desc->ByteSizeOfBlobHeaderField()
        + blob_desc->ByteSizeOfDataIdField());
  } else {
    col_num_ptr_ = nullptr;
  }
  dptr_ = mem_ptr + blob_desc->ByteSizeOfBlobHeaderField()
          + blob_desc->ByteSizeOfDataIdField()
          + blob_desc->ByteSizeOfColNumField();
  blob_desc_ = blob_desc;
  comm_net_token_ = comm_net_token;
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

const void* Blob::memory_ptr() const {
  return reinterpret_cast<void*>(blob_header_);
}

size_t Blob::ByteSizeOfBlobHeaderField() const {
  return blob_desc_->ByteSizeOfBlobHeaderField();
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
void Blob::CopyBlobHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, blob_header_, rhs->blob_header_,
                      ByteSizeOfBlobHeaderField());
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

#define INSTANTIATE_BLOB_FUNC(dev_t)                                       \
  template void Blob::CopyBlobHeaderFrom<dev_t>(DeviceCtx*, const Blob*);  \
  template void Blob::CopyDataContentFrom<dev_t>(DeviceCtx*, const Blob*); \
  template void Blob::CopyDataIdFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyColNumFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyFrom<dev_t>(DeviceCtx*, const Blob*);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BLOB_FUNC, DEVICE_TYPE_SEQ);

}  // namespace oneflow
