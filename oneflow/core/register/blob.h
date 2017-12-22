#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class PieceStatus final {
 public:
  PieceStatus() : piece_id_(0), col_id_(0), max_col_id_(-1) {}
  ~PieceStatus() = default;
  PieceStatus(const PieceStatus&) = default;
  PieceStatus& operator=(const PieceStatus&) = default;

  bool operator==(const PieceStatus& other) const {
    return (piece_id_ == other.piece_id_) && (col_id_ == other.col_id_)
           && (max_col_id_ == other.max_col_id_);
  }
  bool operator!=(const PieceStatus& other) const { return !(*this == other); }

  int64_t piece_id() const { return piece_id_; }
  int64_t col_id() const { return col_id_; }
  int64_t max_col_id() const { return max_col_id_; }

  void set_max_col_id(int64_t max_col_id) {
    CHECK_EQ(-1, max_col_id_);  //-1 for unset
    max_col_id_ = max_col_id;
  }

  int GetIntoNextStatus();
  bool IsLast() const;
  bool IsLastCol() const { return col_id_ == max_col_id_; }
  bool IsNextColOf(const PieceStatus& pre) const;

 private:
  int64_t piece_id_;
  int64_t col_id_;
  int64_t max_col_id_;
};

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const BlobDesc* blob_desc, char* mem_ptr)
      : Blob(blob_desc, mem_ptr, nullptr) {}
  Blob(const BlobDesc* blob_desc, char* mem_ptr, const void* comm_net_token);
  ~Blob() = default;

  const char* data_id(int32_t no) const;
  char* mut_data_id(int32_t no) { return const_cast<char*>(data_id(no)); }

  const char* data_id() const { return data_id(0); }
  char* mut_data_id() { return mut_data_id(0); }

  BlobDesc::OffSetType offset(int32_t no) const;
  BlobDesc::OffSetType& mut_offset(int32_t no);

  BlobDesc::OffSetType offset() const { return offset(0); }
  BlobDesc::OffSetType& mut_offset() { return mut_offset(0); }

  const void* memory_ptr() const {
    if (data_id_ptr_) {
      return static_cast<void*>(data_id_ptr_);
    } else if (offset_ptr_) {
      return static_cast<void*>(offset_ptr_);
    } else {
      return dptr_;
    }
  }
  void* mut_memory_ptr() { return const_cast<void*>(memory_ptr()); }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>();
    return static_cast<const T*>(dptr_);
  }

  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>();
    return static_cast<T*>(dptr_);
  }

  const void* comm_net_token() const { return comm_net_token_; }

  const BlobDesc& blob_desc() const { return *blob_desc_; }
  const BlobDesc* blob_desc_ptr() const { return blob_desc_; }
  const Shape& shape() const { return blob_desc_->shape(); }
  DataType data_type() const { return blob_desc_->data_type(); }
  bool has_data_id() const { return blob_desc_->has_data_id(); }
  size_t ByteSizeOfDataIdField() const {
    return blob_desc_->ByteSizeOfDataIdField();
  }
  size_t ByteSizeOfOffsetField() const {
    return blob_desc_->ByteSizeOfOffsetField();
  }
  size_t ByteSizeOfDataContentField() const {
    return blob_desc_->ByteSizeOfDataContentField();
  }
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  template<DeviceType device_type>
  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyOffSetFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs);

  const PieceStatus& piece_status() const { return piece_status_; }
  void set_piece_status(const PieceStatus& pst) { piece_status_ = pst; }

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false
                   && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::val))
        << blob_desc_->data_type() << " " << GetDataType<T>::val;
  }

  char* data_id_ptr_;
  BlobDesc::OffSetType* offset_ptr_;
  void* dptr_;
  PieceStatus piece_status_;
  const void* comm_net_token_;
  const BlobDesc* blob_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
