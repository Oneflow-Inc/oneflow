#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

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

  int32_t col_num(int32_t no) const;
  void set_col_num(int32_t no, int32_t val);

  const void* memory_ptr() const;
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
  bool has_data_id_field() const { return blob_desc_->has_data_id_field(); }
  bool has_col_num_field() const { return blob_desc_->has_col_num_field(); }
  int32_t max_col_num() const { return blob_desc_->max_col_num(); }
  size_t ByteSizeOfBlobHeaderField() const;
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  template<DeviceType device_type>
  void CopyBlobHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs);
  template<DeviceType device_type>
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs);

  int64_t col_id() const { return blob_header_->col_id; }
  void set_col_id(int64_t val) { blob_header_->col_id = val; }
  int64_t max_col_id() const { return blob_header_->max_col_id; }
  void set_max_col_id(int64_t val) { blob_header_->max_col_id = val; }

  bool IsLastCol() const { return col_id() == max_col_id(); }

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
  int32_t* col_num_ptr_;
  void* dptr_;
  const void* comm_net_token_;
  const BlobDesc* blob_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
