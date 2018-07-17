#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/record_io.h"

namespace oneflow {

class RegstMgr;
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

  const void* hptr() const { return hptr_; }
  void* mut_hptr() { return hptr_; }

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

  void InitOFRecordBlobIfNeed();

  const BlobDesc& blob_desc() const { return *blob_desc_; }
  const BlobDesc* blob_desc_ptr() const { return blob_desc_; }
  const Shape& shape() const { return blob_desc_->shape(); }
  DataType data_type() const { return blob_desc_->data_type(); }
  bool has_data_id_field() const { return blob_desc_->has_data_id_field(); }
  bool has_col_num_field() const { return blob_desc_->has_col_num_field(); }
  int32_t max_col_num() const { return blob_desc_->max_col_num(); }
  size_t ByteSizeOfDataIdField() const { return blob_desc_->ByteSizeOfDataIdField(); }
  size_t ByteSizeOfColNumField() const { return blob_desc_->ByteSizeOfColNumField(); }
  size_t ByteSizeOfHeaderField() const { return blob_desc_->ByteSizeOfHeaderField(); }
  size_t ByteSizeOfDataContentField() const { return blob_desc_->ByteSizeOfDataContentField(); }
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  virtual void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;
  virtual void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) = 0;

  int32_t col_id() const;
  void set_col_id(int32_t val);
  int32_t max_col_id() const;
  void set_max_col_id(int32_t val);
  bool IsColValid() const { return col_id() <= max_col_id(); }
  const MemoryCase& mem_case() const;
  bool IsMemoryContinuous() const;

 protected:
  Blob(Regst* regst, const BlobDesc* blob_desc, char* hptr, char* dptr);

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }

  void* hptr_;
  void* dptr_;
  char* data_id_ptr_;
  int32_t* col_num_ptr_;
  const BlobDesc* blob_desc_;
  Regst* regst_;
};

using CopyBlobFieldMthd = void (Blob::*)(DeviceCtx*, const Blob*);

Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* header_mem_ptr, char* data_mem_ptr,
              DeviceType device_type);
Blob* NewBlob(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr, DeviceType device_type);

void GenOFRecordBlob(Blob* blob);

template<typename RecordType>
class RecordBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordBlob);
  RecordBlob(Blob* records) : records_(records), record_num_(0) {
    CHECK_EQ(records->blob_desc().data_type(), GetDataType<RecordType>::value);
    record_num_ = records_->shape().elem_cnt();
  }
  ~RecordBlob() = default;

  void ForEachRecord(std::function<void(const RecordType&)> Handler) {
    FOR_RANGE(int32_t, i, 0, record_num_) { Handler(*(records_->mut_dptr<RecordType>() + i)); }
  }

  const RecordType& GetRecord(size_t i) {
    CHECK_LT(i, record_num_);
    return *(records_->mut_dptr<RecordType>() + i);
  }

  int32_t record_num() { return record_num_; }

  void ReadFrom(PersistentInStream* in_stream) { record_num_ = ReadRecord(in_stream, records_); }

 private:
  Blob* records_;
  int32_t record_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
