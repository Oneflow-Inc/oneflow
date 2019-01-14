#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/record_io.h"
#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

class RegstMgr;
class Regst;

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  const char* data_id(int32_t no) const;
  char* mut_data_id(int32_t no) { return const_cast<char*>(data_id(no)); }

  const char* data_id() const { return data_id(0); }
  char* mut_data_id() { return mut_data_id(0); }

  int32_t col_num(int32_t no) const;
  void set_col_num(int32_t no, int32_t val);

  const int32_t* col_num() const { return col_num_ptr_; }
  int32_t* mut_col_num() { return col_num_ptr_; }

  const void* header_ptr() const { return header_ptr_; }
  void* mut_header_ptr() { return header_ptr_; }

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

  template<typename T, typename... Int64s>
  typename std::enable_if<!std::is_same<T, void>::value, const T*>::type dptr(
      int64_t dim0, Int64s... remainder_dims) const {
    return dptr<T>() + GetDptrOffset(0, dim0, remainder_dims...);
  }

  template<typename T, typename... Int64s>
  typename std::enable_if<!std::is_same<T, void>::value, T*>::type mut_dptr(
      int64_t dim0, Int64s... remainder_dims) {
    return mut_dptr<T>() + GetDptrOffset(0, dim0, remainder_dims...);
  }

  const RtBlobDesc& blob_desc() const { return *blob_desc_; }
  const RtBlobDesc* blob_desc_ptr() const { return blob_desc_; }
  const Shape& shape() const { return blob_desc_->shape(); }
  DataType data_type() const { return blob_desc_->data_type(); }
  bool has_data_id_field() const { return blob_desc_->has_data_id_field(); }
  bool has_col_num_field() const { return blob_desc_->has_col_num_field(); }
  int32_t max_col_num() const { return blob_desc_->max_col_num(); }
  size_t ByteSizeOfBlobHeader() const { return blob_desc_->ByteSizeOfBlobHeader(); }
  size_t ByteSizeOfDataIdField() const { return blob_desc_->ByteSizeOfDataIdField(); }
  size_t ByteSizeOfColNumField() const { return blob_desc_->ByteSizeOfColNumField(); }
  size_t ByteSizeOfDataContentField() const { return blob_desc_->ByteSizeOfDataContentField(); }
  size_t TotalByteSize() const { return blob_desc_->TotalByteSize(); }

  bool IsContiguous() const { return is_contiguous_; }
  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs);

  const int32_t& record_num() const;
  void set_record_num(int32_t val);
  int32_t col_id() const;
  void set_col_id(int32_t val);
  int32_t max_col_id() const;
  void set_max_col_id(int32_t val);
  bool IsColValid() const { return col_id() <= max_col_id(); }
  const MemoryCase& mem_case() const;
  const PodPtr* header_pod_ptr() const { return &header_pod_ptr_; }
  PodPtr* header_pod_ptr() { return &header_pod_ptr_; }

 private:
  int64_t GetDptrOffset(int32_t index) const { return 0; }
  template<typename... Int64s>
  int64_t GetDptrOffset(int32_t index, int64_t cur_dim, Int64s... remainder) const {
    CHECK_GE(shape().NumAxes(), index + 1);
    CHECK_GE(cur_dim, 0);
    CHECK_LT(cur_dim, shape().At(index));
    return cur_dim * shape().Count(index + 1) + GetDptrOffset(index + 1, remainder...);
  }

  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }
  void Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);

  int32_t record_num_; // FIXME() by dim0
  bool is_contiguous_;
  void* header_ptr_;
  char* data_id_ptr_;
  int32_t* col_num_ptr_;
  void* dptr_;
  const RtBlobDesc* blob_desc_;
  Regst* regst_;
  PodPtr header_pod_ptr_;
};

template<typename RecordType>
class RecordBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordBlob);
  RecordBlob(Blob* records) : records_(records), record_num_(0) {
    CHECK_EQ(records->blob_desc().data_type(), GetDataType<RecordType>::value);
    record_num_ = records_->record_num();
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

  void ReadFrom(PersistentInStream* in_stream) {
    record_num_ = ReadRecord(in_stream, records_);
    records_->set_record_num(record_num_);
  }

 private:
  Blob* records_;
  int32_t record_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
