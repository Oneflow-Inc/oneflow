/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_REGISTER_BLOB_H_
#define ONEFLOW_CORE_REGISTER_BLOB_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class BlobAccessChecker {
 public:
  virtual void CheckHeaderMutable() const = 0;
  virtual void CheckBodyMutable() const = 0;
};

template<bool is_header_mutable, bool is_body_mutable>
class BlobAccessCheckerIf final : public BlobAccessChecker {
 public:
  void CheckHeaderMutable() const override {
    CHECK(is_header_mutable)
        << "header mutable check not passed, blob's shape is not mutable at this moment!";
  }

  void CheckBodyMutable() const override {
    CHECK(is_body_mutable)
        << "body mutable check not passed, blob's data is not mutable at this moment!";
  }
};

class Blob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Blob);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr);
  Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  virtual ~Blob() = default;

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_->ptr(); }
  char* mut_header_ptr() { return header_ptr_->ptr(); }
  char* mut_contiguous_header_ptr();
  const RtBlobDesc& blob_desc() const { return *blob_desc_; }
  const RtBlobDesc* blob_desc_ptr() const { return blob_desc_; }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type());
    return static_cast<const T*>(dptr_);
  }
  template<typename T = void>
  T* mut_dptr() {
    this->blob_access_checker()->CheckBodyMutable();
    CheckDataType<T>(data_type());
    return static_cast<T*>(dptr_);
  }
  template<typename T = void>
  T* ForceMutDptr() {
    CheckDataType<T>(data_type());
    return static_cast<T*>(dptr_);
  }
  const Shape& static_shape() const { return blob_desc_->body_shape(); }
  const ShapeView& shape_view() const { return *shape_view_; }
  const ShapeView& shape() const { return *shape_view_; }
  MutShapeView* mut_shape_view() {
    this->blob_access_checker()->CheckHeaderMutable();
    return mut_shape_view_.get();
  }

  MutShapeView* ForceMutShapeView() { return mut_shape_view_.get(); }

  void reset_dptr(char* dptr) { dptr_ = dptr; }

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs);
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs);
  bool IsBodyEmpty() const { return shape().elem_cnt() == 0; }

  size_t AlignedTotalByteSize() const { return blob_desc_->AlignedTotalByteSize(); }
  const MemoryCase& mem_case() const { return mem_case_; }

  size_t ByteSizeOfBlobBody() const { return blob_desc_->ByteSizeOfBlobBody(); }
  size_t AlignedByteSizeOfBlobBody() const { return blob_desc_->AlignedByteSizeOfBlobBody(); }

  int32_t record_num() const { return record_num_; }
  void set_record_num(int32_t val) { record_num_ = val; }

  void set_blob_access_checker(const BlobAccessChecker* blob_access_checker) {
    this->blob_access_checker_ = blob_access_checker;
  }

  const BlobAccessChecker* blob_access_checker() { return this->blob_access_checker_; }

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

  const BlobAccessChecker* blob_access_checker_;
  MemoryCase mem_case_;
  const RtBlobDesc* blob_desc_;
  void* dptr_;
  int64_t* header_fields_[FieldKey::kFieldKeySize];
  size_t header_field_capacities_[FieldKey::kFieldKeySize];
  std::unique_ptr<ShapeView> shape_view_;
  std::unique_ptr<MutShapeView> mut_shape_view_;
  std::unique_ptr<PodPtr> header_ptr_;
  // TODO(chengcheng); remove record num and record_blob
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

#define INIT_GLOBAL_BLOB_MUTABLE_CHECKER(is_header_mutable, is_body_mutable)             \
  COMMAND(Global<BlobAccessCheckerIf<is_header_mutable, is_body_mutable>>::SetAllocated( \
      new BlobAccessCheckerIf<is_header_mutable, is_body_mutable>()))

INIT_GLOBAL_BLOB_MUTABLE_CHECKER(false, false);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(false, true);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(true, false);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(true, true);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
