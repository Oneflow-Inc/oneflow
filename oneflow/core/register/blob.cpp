#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

int PieceStatus::GetIntoNextStatus() {
  if (IsLast()) { return -1; }
  if (col_id_ == max_col_num_ - 1) {
    piece_id_ += 1;
    col_id_ = 0;
    max_col_num_ = 0;
  } else {
    col_id_ += 1;
  }
  return 0;
}

bool PieceStatus::IsLast() const {
  if (piece_id_ == RuntimeCtx::Singleton()->total_piece_num() - 1
      && col_id_ == max_col_num_ - 1) {
    return true;
  }
  return false;
}

bool PieceStatus::IsNextColOf(const PieceStatus& pre) const {
  if (piece_id_ == pre.piece_id_ && max_col_num_ == pre.max_col_num_
      && col_id_ == pre.col_id_ + 1) {
    return true;
  }
  return false;
}

Blob::Blob(const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token) {
  data_id_ptr_ = blob_desc->has_data_id() ? mem_ptr : nullptr;
  if (blob_desc->has_offset()) {
    offset_ptr_ = reinterpret_cast<BlobDesc::OffSetType*>(
        mem_ptr + blob_desc->ByteSizeOfDataIdField());
  } else {
    offset_ptr_ = nullptr;
  }
  dptr_ = mem_ptr + blob_desc->ByteSizeOfDataIdField()
          + blob_desc->ByteSizeOfOffsetField();
  blob_desc_ = blob_desc;
  comm_net_token_ = comm_net_token;
}

const char* Blob::data_id(int32_t no) const {
  CHECK_NOTNULL(data_id_ptr_);
  return data_id_ptr_ + no * JobDesc::Singleton()->SizeOfOneDataId();
}

BlobDesc::OffSetType Blob::offset(int32_t no) const {
  CHECK_NOTNULL(offset_ptr_);
  return *(offset_ptr_ + no);
}

BlobDesc::OffSetType& Blob::mut_offset(int32_t no) {
  CHECK_NOTNULL(offset_ptr_);
  return *(offset_ptr_ + no);
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
void Blob::CopyOffSetFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, offset_ptr_, rhs->offset_ptr_,
                      ByteSizeOfOffsetField());
}
template<DeviceType device_type>
void Blob::CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  Memcpy<device_type>(device_ctx, mut_memory_ptr(), rhs->memory_ptr(),
                      TotalByteSize());
}

#define INSTANTIATE_BLOB_FUNC(dev_t)                                       \
  template void Blob::CopyDataContentFrom<dev_t>(DeviceCtx*, const Blob*); \
  template void Blob::CopyDataIdFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyOffSetFrom<dev_t>(DeviceCtx*, const Blob*);      \
  template void Blob::CopyFrom<dev_t>(DeviceCtx*, const Blob*);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BLOB_FUNC, DEVICE_TYPE_SEQ);

}  // namespace oneflow
