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
  header_ptr_.reset(new PodPtr(blob_desc_->header_pod_desc(), header_ptr));
  this->blob_access_checker_ = Global<BlobAccessCheckerIf<true, true>>::Get();
  FOR_RANGE(int32_t, i, 0, FieldKey::kFieldKeySize) {
    FieldKey key = static_cast<FieldKey>(i);
    header_fields_[i] = header_ptr_->MutTensorPtr<int64_t>(key);
    if (header_fields_[i] == nullptr) {
      header_field_capacities_[i] = 0;
    } else {
      header_field_capacities_[i] =
          blob_desc->header_pod_desc().Field(key).Cast<TensorPodDesc>().shape().elem_cnt();
    }
  }
  if (!blob_desc_->header_is_opaque()) {
    int64_t* shape_ptr = mut_header_field<FieldKey::kTensorShape>();
    shape_view_.reset(new ShapeView(shape_ptr, static_shape().NumAxes()));
    if (blob_desc->is_dynamic()) {
      mut_shape_view_.reset(new MutShapeView(shape_ptr, static_shape().NumAxes()));
    }
    MutShapeView(shape_ptr, static_shape().NumAxes()).set_shape(static_shape());
  } else {
    const DimVector& dim_vec = static_shape().dim_vec();
    shape_view_.reset(new ShapeView(dim_vec.data(), dim_vec.size()));
  }
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
  if (this == rhs || blob_desc().ByteSizeOfBlobHeader() == 0) { return; }
  CHECK_EQ(blob_desc().ByteSizeOfBlobHeader(), rhs->blob_desc().ByteSizeOfBlobHeader());
  if (blob_desc().header_is_opaque()) {
    Memcpy<DeviceType::kCPU>(device_ctx, header_ptr_->ptr(), rhs->header_ptr(),
                             blob_desc().ByteSizeOfBlobHeader());
    return;
  }
  {
    const size_t num_axes = static_shape().NumAxes();
    Memcpy<DeviceType::kCPU>(device_ctx, mut_header_field<FieldKey::kTensorShape>(),
                             rhs->header_field<FieldKey::kTensorShape>(),
                             num_axes * sizeof(int64_t));
  }
}

char* Blob::mut_contiguous_header_ptr() {
  // check header and body is continuous
  CHECK_EQ(header_ptr() + blob_desc_->ByteSizeOfBlobHeader(), dptr<char>());
  return header_ptr_->ptr();
}

}  // namespace oneflow
