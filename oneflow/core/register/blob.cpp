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
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr) {
  Init(mem_case, blob_desc, header_ptr, header_ptr + blob_desc->ByteSizeOfBlobHeader());
}

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
           char* body_ptr) {
  Init(mem_case, blob_desc, header_ptr, body_ptr);
}

void Blob::Init(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
                char* body_ptr) {
  mem_case_ = mem_case;
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
  header_ptr_ = header_ptr;
  this->blob_access_checker_ = Global<BlobAccessCheckerIf<true, true>>::Get();
  int64_t* shape_ptr = reinterpret_cast<int64_t*>(header_ptr);
  shape_view_.reset(new ShapeView(shape_ptr, static_shape().NumAxes()));
  if (blob_desc->is_dynamic()) {
    mut_shape_view_.reset(new MutShapeView(shape_ptr, static_shape().NumAxes()));
  }
  MutShapeView(shape_ptr, static_shape().NumAxes()).set_shape(static_shape());
}

void Blob::CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  this->blob_access_checker()->CheckBodyMutable();
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfBlobBody(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  this->blob_access_checker()->CheckBodyMutable();
  const size_t body_byte_size = ByteSizeOfBlobBody();
  CHECK_EQ(rhs->ByteSizeOfBlobBody(), body_byte_size);
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), body_byte_size, mem_case(), rhs->mem_case());
}

void Blob::CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  size_t header_size = blob_desc().ByteSizeOfBlobHeader();
  if (this == rhs || header_size == 0) { return; }
  CHECK_EQ(header_size, rhs->blob_desc().ByteSizeOfBlobHeader());
  Memcpy<DeviceType::kCPU>(device_ctx, header_ptr_, rhs->header_ptr(), header_size);
}

char* Blob::mut_contiguous_header_ptr() {
  // check header and body is continuous
  CHECK_EQ(header_ptr() + blob_desc_->ByteSizeOfBlobHeader(), dptr<char>());
  return header_ptr_;
}

}  // namespace oneflow
