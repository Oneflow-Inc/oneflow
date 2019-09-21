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

class DenseShapeWrapper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DenseShapeWrapper);
  DenseShapeWrapper() = default;
  ~DenseShapeWrapper() = default;

  void Init(int64_t* ptr, int64_t dense_shape_num_axes) {
    CHECK_NOTNULL(ptr);
    ptr_ = ptr;
    num_axes_ = dense_shape_num_axes;
    is_shape_inited_ = false;
  }

  const Shape& shape() const {
    CHECK(is_shape_inited_);
    return shape_;
  }
  void set_shape(const Shape& val) {
    CHECK_EQ(num_axes_, val.NumAxes());
    shape_ = val;
    is_shape_inited_ = true;
    for (size_t i = 0; i < shape_.NumAxes(); ++i) { *(ptr_ + i) = shape_.At(i); }
  }

 private:
  int64_t* ptr_;
  int64_t num_axes_;

  Shape shape_;
  bool is_shape_inited_;
};

class LoDWrapper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LoDWrapper);
  ~LoDWrapper() = default;

  void Init(int64_t* ptr, int64_t max_reserved_size_for_lod, int64_t num_of_lod_levels) {
    CHECK_NOTNULL(ptr);
    ptr_ = ptr;
    max_reserved_size_for_lod_ = max_reserved_size_for_lod;
    offset_lod_.resize(num_of_lod_levels);
    for (std::vector<int64_t>& vec : offset_lod_) { vec.push_back(0); }
    lod_cnt_ = 0;
    is_lod_done_ = false;
  }

  int64_t GetOffset(int64_t level, int64_t pos) {
    CHECK(is_lod_done_);
    return offset_lod_.at(level).at(pos);
  }
  int64_t GetLength(int64_t level, int64_t pos) {
    return GetOffset(level, pos + 1) - GetOffset(level, pos);
  }
  void PushLength(int64_t level, int64_t len) {
    is_lod_done_ = false;
    if (lod_cnt_ + 1 + offset_lod_.size() > max_reserved_size_for_lod_) {
      LOG(FATAL) << "the LoD size is greater than max_reserved_size_for_lod: "
                 << max_reserved_size_for_lod_;
    }
    int64_t offset_of_len = offset_lod_.at(level).back() + len;
    offset_lod_.at(level).push_back(offset_of_len);
    lod_cnt_ += 1;
  }
  void SetLoDDone() {
    is_lod_done_ = true;
    size_t i = 0;
    for (const std::vector<int64_t>& vec : offset_lod_) {
      for (int64_t offset : vec) {
        *(ptr_ + i) = offset;
        i += 1;
      }
    }
  }

 private:
  int64_t* ptr_;
  int64_t max_reserved_size_for_lod_;

  std::vector<std::vector<int64_t>> offset_lod_;
  int64_t lod_cnt_;  // attention: no equals to the elem_cnt of offset_lod_
  bool is_lod_done_;
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
  const Shape& static_shape() const { blob_desc_->body_shape(); }
  const Shape& shape() const {
    return dense_shape(); // TODO(niuchong): remove this interface
  }
  const Shape& dense_shape() const { dense_shape_.shape(); }
  void set_dense_shape(const Shape& shape) { dense_shape_.set_shape(shape); }

  int64_t GetLoDOffset(int64_t level, int64_t pos) { lod_.GetOffset(level, pos); }
  int64_t GetLoDLength(int64_t level, int64_t pos) { lod_.GetLength(level, pos); }
  void PushLoDLength(int64_t level, int64_t len) { lod_.PushLength(level, len); }
  void SetLoDDone() { lod_.SetLoDDone(); }

  size_t AlignedTotalByteSize(size_t align) const {
    return blob_desc_->AlignedTotalByteSize(align);
  }
  const MemoryCase& mem_case() const;

 private:
  void Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && blob_desc_->data_type() != DataType::kChar
                   && blob_desc_->data_type() != GetDataType<T>::value))
        << blob_desc_->data_type() << " " << GetDataType<T>::value;
  }
  int64_t GetDptrOffset(int32_t index) const { return 0; }
  template<typename... Int64s>
  int64_t GetDptrOffset(int32_t index, int64_t cur_dim, Int64s... remainder) const {
    CHECK_GE(static_shape().NumAxes(), index + 1);
    CHECK_GE(cur_dim, 0);
    CHECK_LT(cur_dim, static_shape().At(index));
    return cur_dim * static_shape().Count(index + 1) + GetDptrOffset(index + 1, remainder...);
  }

  Regst* regst_;
  bool is_contiguous_;

  const RtBlobDesc* blob_desc_;
  void* dptr_;
  PodPtr header_ptr_;

  DenseShapeWrapper dense_shape_;
  LoDWrapper lod_;
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
