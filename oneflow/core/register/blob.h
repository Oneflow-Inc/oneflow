#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

struct BlobHeader {
  BlobHeader() : piece_id(-1), col_id(-1), max_col_num(-1) {}

  int64_t piece_id;
  int64_t col_id;
  int64_t max_col_num;
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

  int32_t seq_len(int32_t no) const;
  int32_t* mut_seq_len(int32_t no);

  int32_t seq_len() const { return seq_len(0); }
  int32_t* mut_seq_len() { return mut_seq_len(0); }

  const void* memory_ptr() const {
    return reinterpret_cast<void*>(blob_header_);
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
  bool has_seq_len() const { return blob_desc_->has_seq_len(); }
  size_t ByteSizeOfBlobHeaderField() const {
    return blob_desc_->ByteSizeOfBlobHeaderField();
  }
  size_t ByteSizeOfDataIdField() const {
    return blob_desc_->ByteSizeOfDataIdField();
  }
  size_t ByteSizeOfSeqLenField() const {
    return blob_desc_->ByteSizeOfSeqLenField();
  }
  size_t ByteSizeOfDataContentField() const {
    return blob_desc_->ByteSizeOfDataContentField();
  }
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  template<DeviceType device_type>
  void CopyBlobHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopySeqLenFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs);

  int64_t piece_id() const { return blob_header_->piece_id; }
  int64_t col_id() const { return blob_header_->col_id; }
  int64_t max_col_num() const { return blob_header_->max_col_num; }

  void set_piece_id(int64_t val) { blob_header_->piece_id = val; }
  void set_col_id(int64_t val) {
    CHECK_LT(val, blob_header_->max_col_num);
    blob_header_->col_id = val;
  }
  void set_max_col_num(int64_t val) { blob_header_->max_col_num = val; }

  bool HasSamePieceStatus(const Blob& other) const {
    return piece_id() == other.piece_id() && col_id() == other.col_id()
           && max_col_num() == other.max_col_num();
  }
  bool IsLastCol() const {
    CHECK_NE(-1, piece_id());
    return col_id() == max_col_num() - 1;
  }
  bool IsNextColOf(const Blob& pre) const {
    CHECK_NE(-1, piece_id());
    return piece_id() == pre.piece_id() && max_col_num() == pre.max_col_num()
           && col_id() == pre.col_id() + 1;
  }

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false
                   && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::val))
        << blob_desc_->data_type() << " " << GetDataType<T>::val;
  }

  BlobHeader* blob_header_;
  char* data_id_ptr_;
  int32_t* seq_len_ptr_;
  void* dptr_;
  const void* comm_net_token_;
  const BlobDesc* blob_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
