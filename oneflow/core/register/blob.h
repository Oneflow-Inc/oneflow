#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/record_io.h"

namespace oneflow {

class Regst;

class BlobIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobIf);
  virtual ~BlobIf() = default;

 protected:
  BlobIf() = default;
};

class Blob : public BlobIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  virtual ~Blob() = default;

  const char* data_id(int32_t no) const;
  char* mut_data_id(int32_t no) { return const_cast<char*>(data_id(no)); }

  const char* data_id() const { return data_id(0); }
  char* mut_data_id() { return mut_data_id(0); }

  int32_t col_num(int32_t no) const;
  void set_col_num(int32_t no, int32_t val);

  const int32_t* col_num() const { return col_num_ptr_; }
  int32_t* mut_col_num() { return col_num_ptr_; }

  const void* memory_ptr() const { return mem_ptr_; }
  void* mut_memory_ptr() { return mem_ptr_; }

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
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  virtual void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;

  int32_t col_id() const;
  void set_col_id(int32_t val);
  int32_t max_col_id() const;
  void set_max_col_id(int32_t val);
  bool IsColValid() const;

 protected:
  Blob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr)
      : Blob(regst, blob_desc, mem_ptr, nullptr) {}
  Blob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
       const void* comm_net_token);

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false
                   && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }

  void* mem_ptr_;
  char* data_id_ptr_;
  int32_t* col_num_ptr_;
  void* dptr_;
  const void* comm_net_token_;
  const BlobDesc* blob_desc_;
  Regst* regst_;
};

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
              const void* comm_net_token, DeviceType device_type);

class RecordBlobIf : public BlobIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordBlobIf);
  RecordBlobIf() = default;
  virtual ~RecordBlobIf() = default;

  virtual void ReadFrom(PersistentInStream* in_stream) = 0;
  virtual int32_t record_num() = 0;

 private:
};

template<typename RecordType>
class RecordBlob final : public RecordBlobIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordBlob);
  RecordBlob()
      : records_(Global<JobDesc>::Get()->SinglePieceSize()), record_num_(0) {}
  ~RecordBlob() = default;

  void ForEachRecord(std::function<void(const RecordType&)> Handler) {
    FOR_RANGE(int32_t, i, 0, record_num_) { Handler(records_.at(i)); }
  }

  RecordType* mut_records(size_t i) {
    CHECK_LT(i, record_num_);
    return &(records_.at(i));
  }

  int32_t record_num() override { return record_num_; }

  void ReadFrom(PersistentInStream* in_stream) override {
    record_num_ = ReadRecord<RecordType>(in_stream, &records_);
  }

 private:
  std::vector<RecordType> records_;
  int32_t record_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
