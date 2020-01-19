#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/register/tensor_view.h"
#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Blob;

class TensorBackInserter final {
 public:
  explicit TensorBackInserter(Blob* blob);
  TensorBackInserter(const TensorBackInserter&) = default;
  ~TensorBackInserter() = default;

  void ReserveOneEmptyTensorList();
  void ClearTensorLists();
  bool IsCurMutTensorAvailable() const;
  FullyMutTensorView* add_tensor();
  FullyMutTensorView* cur_mut_tensor();
  void add_tensor_list_slice();

 private:
  Blob* blob_;
  FullyMutTensorView cur_mut_tensor_;
};

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  // tensor view for blob
  const TensorView& sole_tensor() const;
  const DataOnlyMutTensorView& sole_mut_tensor();

  // [tensor] view for blob
  size_t total_num_of_tensors() const;
  const TensorView& BeginTensor() const { return *begin_tensor_; }
  const DataOnlyMutTensorView& BeginMutTensor() { return *begin_mut_tensor_; }
  void MoveToNextTensor(TensorView* last) const;
  void MoveToNextMutTensor(DataOnlyMutTensorView* last);
  bool IsEndTensor(const TensorView& tensor) const;
  bool IsEndTensor(const DataOnlyMutTensorView& tensor) const;

  friend class TensorBackInserter;

  // tensor list slice
  size_t num_of_tensor_list_slices() const;
  int64_t tensor_index4slice_id(int32_t slice_id) const;

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_->ptr(); }
  const RtBlobDesc& blob_desc() const { return *blob_desc_; }
  const RtBlobDesc* blob_desc_ptr() const { return blob_desc_; }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type());
    return static_cast<const T*>(dptr_);
  }
  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>(data_type());
    return static_cast<T*>(dptr_);
  }
  const Shape& static_shape() const { return blob_desc_->body_shape(); }
  const ShapeView& shape_view() const { return *shape_view_; }
  const ShapeView& shape() const { return *shape_view_; }
  MutShapeView* mut_shape_view() { return mut_shape_view_.get(); }

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  bool IsBodyEmpty() const;

  size_t AlignedTotalByteSize() const { return blob_desc_->AlignedTotalByteSize(); }
  const MemoryCase& mem_case() const;

  size_t ByteSizeOfBlobBody() const;
  size_t AlignedByteSizeOfBlobBody() const { return blob_desc_->AlignedByteSizeOfBlobBody(); }

  int32_t record_num() const { return record_num_; }
  void set_record_num(int32_t val) { record_num_ = val; }

 private:
  void Init(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
            char* body_ptr);
  template<FieldKey key>
  const int64_t* header_field() const {
    return header_fields_[key];
  }
  template<FieldKey key>
  int64_t* mut_header_field() {
    return header_fields_[key];
  }
  template<FieldKey key>
  size_t header_field_capacity() const {
    return header_field_capacities_[key];
  }
  size_t GetEndTensorDataOffset() const;

  FullyMutTensorView EndFullyMutTensor();
  void ReserveOneEmptyTensorList();
  void AddTensor(FullyMutTensorView* tensor);
  bool IsEndFullyMutTensor(const FullyMutTensorView& tensor) const;
  void clear_tensor_lists();
  void add_tensor_list_slice();

  MemoryCase mem_case_;
  const RtBlobDesc* blob_desc_;
  void* dptr_;
  int64_t* header_fields_[FieldKey::kFieldKeySize];
  size_t header_field_capacities_[FieldKey::kFieldKeySize];
  std::unique_ptr<ShapeView> shape_view_;
  std::unique_ptr<MutShapeView> mut_shape_view_;
  std::unique_ptr<PodPtr> header_ptr_;
  std::unique_ptr<TensorView> begin_tensor_;
  std::unique_ptr<DataOnlyMutTensorView> begin_mut_tensor_;
  // TODO(); remove this ugly code
  int32_t record_num_;
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

 private:
  Blob* records_;
  int32_t record_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
