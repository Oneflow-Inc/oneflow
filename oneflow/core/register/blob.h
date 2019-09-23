#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

class RegstMgr;
class Regst;

class DenseShapeViewBase {
 protected:
  DenseShapeViewBase(PodPtr dense_shape_ptr);
  DenseShapeViewBase(const DenseShapeViewBase& rhs) : ptr_(rhs.ptr_), num_axes_(rhs.num_axes_) {}
  virtual ~DenseShapeViewBase() = default;

  int64_t* ptr_;
  int64_t num_axes_;
};

class DenseShapeView final : public DenseShapeViewBase {
 public:
  DenseShapeView(const PodPtr& dense_shape_ptr) : DenseShapeViewBase(dense_shape_ptr) {}
  DenseShapeView(const DenseShapeView& rhs) : DenseShapeViewBase(rhs) {}
  ~DenseShapeView() = default;

  int64_t NumAxes() const { return num_axes_; }
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t elem_cnt() const;

  bool operator==(const DenseShapeView& rhs) const;
  std::string ToString() const;

  operator Shape() const;
};

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape);

class DenseShapeMutView final : public DenseShapeViewBase {
 public:
  DenseShapeMutView(PodPtr dense_shape_ptr) : DenseShapeViewBase(dense_shape_ptr) {}
  DenseShapeMutView(const DenseShapeView& rhs) : DenseShapeViewBase(rhs) {}
  ~DenseShapeMutView() = default;

  void set_shape(const Shape& val);
};

class LoDViewBase {
 protected:
  typedef std::vector<std::vector<int64_t>> LoDVec;

  LoDViewBase(PodPtr lod_ptr, int64_t num_of_lod_levels);
  LoDViewBase(const LoDViewBase& rhs) { *this = rhs; }
  LoDViewBase& operator=(const LoDViewBase& rhs);
  ~LoDViewBase() = default;

  LoDVec InitOffsetVecFromPtr() const;
  void FlushOffsetVecToPtr(const LoDVec& offset_lod_vec);

  LoDVec GetLengthLoDVecFromOffsetLoDVec(const LoDVec& offset_lod_vec) const;
  LoDVec GetOffsetLoDVecFromLengthLoDVec(const LoDVec& length_lod_vec) const;

  int64_t* ptr_;
  int64_t num_of_lod_levels_;
  int64_t max_reserved_size_for_lod_;
};

class OffsetLoDView final : public LoDViewBase {
 public:
  OffsetLoDView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels), offset_lod_vec_() {}
  OffsetLoDView(const OffsetLoDView& rhs)
      : LoDViewBase(rhs), offset_lod_vec_(rhs.offset_lod_vec_) {}

  int64_t GetOffset(size_t level, size_t pos);

 private:
  LoDVec offset_lod_vec_;
};

class OffsetLoDMutView final : public LoDViewBase {
 public:
  OffsetLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  OffsetLoDMutView(const OffsetLoDMutView& rhs) : LoDViewBase(rhs) {}

  void SetOffset(const LoDVec& offset_lod_vec);
};

class LengthLoDView final : public LoDViewBase {
 public:
  LengthLoDView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels), length_lod_vec_() {}
  LengthLoDView(const LengthLoDView& rhs) : LoDViewBase(rhs) {}

  int64_t GetLength(size_t level, size_t pos);

 private:
  LoDVec length_lod_vec_;
};

class LengthLoDMutView final : public LoDViewBase {
 public:
  LengthLoDMutView(const PodPtr& lod_ptr, int64_t num_of_lod_levels)
      : LoDViewBase(lod_ptr, num_of_lod_levels) {}
  LengthLoDMutView(const LengthLoDMutView& rhs) : LoDViewBase(rhs) {}

  void SetLength(const LoDVec& length_lod_vec);
};

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_.ptr(); }
  char* mut_header_ptr() { return header_ptr_.ptr(); }
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
    return DenseShapeView(header_ptr_.Field(FieldKey::kDenseShape));
  }
  DenseShapeView shape() const { return dense_shape_view(); }
  DenseShapeMutView dense_shape_mut_view() {
    return DenseShapeMutView(header_ptr_.MutField(FieldKey::kDenseShape));
  }
  LengthLoDView length_lod_view() const {
    return LengthLoDView(header_ptr_.Field(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  LengthLoDMutView length_lod_mut_view() {
    return LengthLoDMutView(header_ptr_.MutField(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  OffsetLoDView offset_lod_view() const {
    return OffsetLoDView(header_ptr_.Field(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }
  OffsetLoDMutView offset_lod_mut_view() {
    return OffsetLoDMutView(header_ptr_.MutField(FieldKey::kLoD), blob_desc_->num_of_lod_levels());
  }

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  bool IsShapeEmpty() const { return false; }

  size_t AlignedTotalByteSize() const { return blob_desc_->AlignedTotalByteSize(); }
  const MemoryCase& mem_case() const;

  // legacy interface, shouldn't use in new code
  size_t ByteSizeOfDataContentField() const { return blob_desc_->ByteSizeOfBlobBody(); }

  int32_t record_num() const { return record_num_; }
  void set_record_num(int32_t val) { record_num_ = val; }

 private:
  void Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }

  bool is_header_body_contiguous_;

  const RtBlobDesc* blob_desc_;
  void* dptr_;
  PodPtr header_ptr_;

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
