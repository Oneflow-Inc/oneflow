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
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

Blob::Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr) {
  Init(mem_case, blob_desc, header_ptr, nullptr, 0);
}

Blob::Blob(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr,
           char* body_ptr) {
  Init(mem_case, blob_desc, header_ptr, body_ptr, 0);
}

Blob::Blob(const MemoryCase& mem_case,  // NOLINT，Blob::Blob(...) { // NOLINT
           const BlobDesc* blob_desc, char* header_ptr, char* body_ptr, const int64_t offset) {
  Init(mem_case, blob_desc, header_ptr, body_ptr, offset);
}

void Blob::Init(const MemoryCase& mem_case, const BlobDesc* blob_desc, char* header_ptr,
                char* body_ptr, const int64_t offset) {
  mem_case_ = mem_case;
  blob_desc_ = blob_desc;
  storage_offset_ = offset;
  dptr_ = body_ptr;
  header_ptr_ = header_ptr;
  this->blob_access_checker_ = Singleton<BlobAccessCheckerIf<true, true>>::Get();
  int64_t* shape_ptr = reinterpret_cast<int64_t*>(header_ptr);
  shape_view_.reset(new ShapeView(shape_ptr, static_shape().NumAxes()));
  if (blob_desc->is_dynamic()) {
    mut_shape_view_.reset(new MutShapeView(shape_ptr, static_shape().NumAxes()));
  }
  MutShapeView(shape_ptr, static_shape().NumAxes()).set_shape(static_shape());
}

void Blob::CopyHeaderFrom(const Blob* rhs) {
  size_t header_size = blob_desc().ByteSizeOfBlobHeader();
  CHECK_EQ(header_size, rhs->blob_desc().ByteSizeOfBlobHeader());
  if (this == rhs || header_size == 0) { return; }
  std::memcpy(header_ptr_, rhs->header_ptr(), header_size);
}

char* Blob::mut_contiguous_header_ptr() {
  // check header and body is continuous
  CHECK_EQ(header_ptr() + blob_desc_->AlignedByteSizeOfBlobHeader(), dptr<char>());
  return header_ptr_;
}

}  // namespace oneflow
