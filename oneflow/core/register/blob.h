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
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/common/shape_view.h"
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
  Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr);
  Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr, char* body_ptr);
  Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr, char* body_ptr,
       const int64_t offset);
  virtual ~Blob() = default;

  DataType data_type() const { return blob_desc_->data_type(); }
  const char* header_ptr() const { return header_ptr_; }
  [[deprecated(
      "\"mut_header_ptr\" will be removed in Bolb. Please avoid to use this method whenever "
      "possible. Almost all methods of `mut_header_ptr` are also in `Blob`.")]] char*
  mut_header_ptr() {
    return header_ptr_;
  }
  char* mut_contiguous_header_ptr();
  const BlobDesc& blob_desc() const { return *blob_desc_; }
  const BlobDesc* blob_desc_ptr() const { return blob_desc_; }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type());
    return reinterpret_cast<T*>(static_cast<char*>(dptr_)
                                + storage_offset_ * GetSizeOfDataType(data_type()));
  }
  template<typename T = void>
  T* mut_dptr() {
    this->blob_access_checker()->CheckBodyMutable();
    CheckDataType<T>(data_type());
    return reinterpret_cast<T*>(static_cast<char*>(dptr_)
                                + storage_offset_ * GetSizeOfDataType(data_type()));
  }
  template<typename T = void>
  T* ForceMutDptr() {
    CheckDataType<T>(data_type());
    return reinterpret_cast<T*>(static_cast<char*>(dptr_)
                                + storage_offset_ * GetSizeOfDataType(data_type()));
  }
  template<typename T = void>
  const T* raw_dptr() const {
    CheckDataType<T>(data_type());
    return static_cast<T*>(dptr_);
  }
  template<typename T = void>
  T* mut_raw_dptr() {
    this->blob_access_checker()->CheckBodyMutable();
    CheckDataType<T>(data_type());
    return static_cast<T*>(dptr_);
  }

  // shape
  const Shape& static_shape() const { return blob_desc_->shape(); }
  const ShapeView& shape_view() const { return *shape_view_; }
  const ShapeView& shape() const { return *shape_view_; }
  MutShapeView* mut_shape_view() {
    this->blob_access_checker()->CheckHeaderMutable();
    return mut_shape_view_.get();
  }
  MutShapeView* ForceMutShapeView() { return mut_shape_view_.get(); }

  // stride
  const Stride& stride() const { return blob_desc_->stride(); }

  void reset_dptr(char* dptr) { dptr_ = dptr; }

  void CopyHeaderFrom(const Blob* rhs);
  bool IsBodyEmpty() const { return shape().elem_cnt() == 0; }

  size_t AlignedTotalByteSize() const { return blob_desc_->AlignedTotalByteSize(); }
  const MemoryCase& mem_case() const { return mem_case_; }

  size_t ByteSizeOfBlobBody() const { return blob_desc_->ByteSizeOfBlobBody(); }
  size_t AlignedByteSizeOfBlobBody() const { return blob_desc_->AlignedByteSizeOfBlobBody(); }

  void set_blob_access_checker(const BlobAccessChecker* blob_access_checker) {
    this->blob_access_checker_ = blob_access_checker;
  }

  const BlobAccessChecker* blob_access_checker() { return this->blob_access_checker_; }

 private:
  void Init(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr, char* body_ptr,
            const int64_t offset);

  const BlobAccessChecker* blob_access_checker_;
  MemoryCase mem_case_;
  const BlobDesc* blob_desc_;
  void* dptr_;
  char* header_ptr_;
  int64_t storage_offset_;
  std::unique_ptr<ShapeView> shape_view_;
  std::unique_ptr<MutShapeView> mut_shape_view_;
};

#define INIT_GLOBAL_BLOB_MUTABLE_CHECKER(is_header_mutable, is_body_mutable)                \
  COMMAND(Singleton<BlobAccessCheckerIf<is_header_mutable, is_body_mutable>>::SetAllocated( \
      new BlobAccessCheckerIf<is_header_mutable, is_body_mutable>()))

INIT_GLOBAL_BLOB_MUTABLE_CHECKER(false, false);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(false, true);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(true, false);
INIT_GLOBAL_BLOB_MUTABLE_CHECKER(true, true);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_H_
