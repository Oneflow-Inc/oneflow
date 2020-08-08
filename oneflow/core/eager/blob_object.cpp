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
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {
namespace eager {

Maybe<void> BlobObject::TryInitBlob() {
  if (!blob_) { JUST(InitBlob()); }
  return Maybe<void>::Ok();
}

Maybe<void> BlobObject::InitBlob() {
  CHECK_NE_OR_RETURN(blob_desc_.data_type(), DataType::kInvalidDataType);
  rt_blob_desc_.reset(new RtBlobDesc(blob_desc_));
  {
    header_buffer_.reset();
    int64_t header_byte_size = rt_blob_desc_->ByteSizeOfBlobHeader();
    const auto& FreeHeader = [header_byte_size](char* dptr) { std::free(dptr); };
    char* ptr = reinterpret_cast<char*>(std::malloc(header_byte_size));
    header_buffer_ = std::unique_ptr<char, std::function<void(char*)>>(ptr, FreeHeader);
  }
  blob_.reset(new Blob(*mem_case_, rt_blob_desc_.get(), header_buffer_.get(), nullptr));
  return Maybe<void>::Ok();
}

Maybe<void> BlobObject::CheckMemCase(const ParallelDesc& parallel_desc, int64_t machine_id) const {
  CHECK_OR_RETURN(parallel_desc.HasMachineId(machine_id))
      << "ParallelDesc does not contain machine_id: " << machine_id;
  const char* device_tag = JUST(DeviceTag4DeviceType(parallel_desc.device_type()));
  if (parallel_desc.device_type() == DeviceType::kCPU) {
    CHECK_OR_RETURN(this->mem_case_->has_host_mem())
        << "DeviceType: " << device_tag
        << " not match MemoryCase: " << this->mem_case_->host_mem().DebugString();
  } else if (parallel_desc.device_type() == DeviceType::kGPU) {
    CHECK_OR_RETURN(this->mem_case_->has_device_cuda_mem())
        << "DeviceType: " << device_tag
        << " not match MemoryCase: " << this->mem_case_->device_cuda_mem().DebugString();
    CHECK_OR_RETURN(
        parallel_desc.Containing(machine_id, this->mem_case_->device_cuda_mem().device_id()));
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

void BlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL(allocator);
  Blob* blob = mut_blob();
  CHECK_NOTNULL(blob);
  const std::size_t required_body_bytes = blob->AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL(blob->dptr());
    return;
  }
  if (blob->dptr() != nullptr) {
    CHECK_EQ(blob_body_bytes_, required_body_bytes);
    return;
  }
  {
    // reset blob_dptr_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
      allocator->Deallocate(dptr, required_body_bytes);
    };
    char* dptr = nullptr;
    blob_dptr_.reset();
    allocator->Allocate(&dptr, required_body_bytes);
    blob_dptr_ = std::unique_ptr<char, std::function<void(char*)>>(dptr, Free);
    blob->reset_dptr(dptr);
    InitNonPODTypeBlobIfNeed(&non_pod_initer_, blob_.get());
  }
  blob_body_bytes_ = required_body_bytes;
}

}  // namespace eager
}  // namespace oneflow
