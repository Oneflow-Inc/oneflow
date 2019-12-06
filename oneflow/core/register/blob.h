#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/register/dense_shape_view.h"
#include "oneflow/core/register/tensor_view.h"
#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Range;

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  // tensor view for blob
  const TensorView& sole_tensor() const;
  DataOnlyMutTensorView* sole_mut_tensor();

  // [tensor] view for blob
  size_t total_num_of_tensors() const;
  std::unique_ptr<TensorView> first_tensor() const;
  std::unique_ptr<TensorView> next_tensor(const TensorView& last) const;
  std::unique_ptr<DataOnlyMutTensorView> first_mut_tensor();
  std::unique_ptr<DataOnlyMutTensorView> next_mut_tensor(const DataOnlyMutTensorView& last);
  std::unique_ptr<FullyMutTensorView> add_tensor(const FullyMutTensorView* last);

  // tensor list slice
  size_t num_of_tensor_list_slices() const;
  int64_t tensor_index4slice_id(int32_t slice_id) const;
  void add_tensor_list_slice();

  // [[tensor]] view for blob
  std::unique_ptr<TensorListView> tensor_list(int32_t slice_id) const;
  std::unique_ptr<MutTensorListView> mut_tensor_list(int32_t slice_id);
  void clear_tensor_lists();

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_->ptr(); }
  const RtBlobDesc& blob_desc() const { return *blob_desc_; }
  const RtBlobDesc* blob_desc_ptr() const { return blob_desc_; }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(blob_desc_->data_type());
    return static_cast<const T*>(dptr_);
  }
  template<typename T = void>
  T* mut_dptr() {
    CheckDataType<T>(blob_desc_->data_type());
    return static_cast<T*>(dptr_);
  }
  const Shape& static_shape() const { return blob_desc_->body_shape(); }
  const DenseShapeView& dense_shape_view() const { return *dense_shape_view_; }
  const DenseShapeView& shape() const { return *dense_shape_view_; }
  DenseShapeMutView* dense_shape_mut_view() { return dense_shape_mut_view_.get(); }

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
  const Shape& header_field_shape(FieldKey field_key) const;
  const int64_t* header_field(FieldKey field_key) const;
  int64_t* mut_header_field(FieldKey field_key);
  void GetTensorListSliceRange(int32_t slice_id, Range* range) const;
  void GetTensorListInfoBySliceId(int32_t slice_id, Range* range, int64_t* shape_offset,
                                  int64_t* data_offset) const;

  MemoryCase mem_case_;
  const RtBlobDesc* blob_desc_;
  void* dptr_;
  std::unique_ptr<DenseShapeView> dense_shape_view_;
  std::unique_ptr<DenseShapeMutView> dense_shape_mut_view_;
  std::unique_ptr<PodPtr> header_ptr_;
  std::unique_ptr<TensorView> sole_tensor_;
  std::unique_ptr<DataOnlyMutTensorView> sole_mut_tensor_;
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
