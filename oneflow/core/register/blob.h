#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/register/dense_shape_view.h"
#include "oneflow/core/register/lod_view.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_->ptr(); }
  char* mut_header_ptr() { return header_ptr_->ptr(); }
  const RtBlobDesc& blob_desc() const { return *blob_desc_; }
  const RtBlobDesc* blob_desc_ptr() const { return blob_desc_; }

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
  const Shape& static_shape() const { return blob_desc_->body_shape(); }
  DenseShapeView dense_shape_view() const {
    return DenseShapeView(header_ptr_->Field(FieldKey::kDenseShape));
  }
  Shape shape() const {
    if (blob_desc().header_is_opaque()) {
      return static_shape();
    } else {
      return dense_shape_view();
    }
  }
  DenseShapeMutView dense_shape_mut_view() {
    return DenseShapeMutView(header_ptr_->MutField(FieldKey::kDenseShape));
  }
  LengthLoDView length_lod_view() const {
    return LengthLoDView(header_ptr_->Field(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  LengthLoDMutView length_lod_mut_view() {
    return LengthLoDMutView(header_ptr_->MutField(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  OffsetLoDView offset_lod_view() const {
    return OffsetLoDView(header_ptr_->Field(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  OffsetLoDMutView offset_lod_mut_view() {
    return OffsetLoDMutView(header_ptr_->MutField(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  TreeLoDView tree_lod_view() const {
    return TreeLoDView(header_ptr_->Field(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  TreeLoDMutView tree_lod_mut_view() const {
    return TreeLoDMutView(header_ptr_->MutField(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  CoordinateLoDMutView coord_lod_mut_view() const {
    return CoordinateLoDMutView(header_ptr_->MutField(FieldKey::kLoD),
                                blob_desc_->num_of_lod_levels());
  }

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  bool IsShapeEmpty() const { return false; }

  size_t AlignedTotalByteSize() const { return blob_desc_->AlignedTotalByteSize(); }
  const MemoryCase& mem_case() const;

  size_t ByteSizeOfBlobBody() const { return blob_desc_->ByteSizeOfBlobBody(); }
  size_t AlignedByteSizeOfBlobBody() const { return blob_desc_->AlignedByteSizeOfBlobBody(); }

  int32_t record_num() const { return record_num_; }
  void set_record_num(int32_t val) { record_num_ = val; }

 private:
  void Init(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
            char* body_ptr);
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }

  MemoryCase mem_case_;
  bool is_header_body_contiguous_;

  const RtBlobDesc* blob_desc_;
  void* dptr_;
  std::unique_ptr<PodPtr> header_ptr_;

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
