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
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {
namespace vm {

EagerBlobObject::EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                                 const std::shared_ptr<Shape>& shape, DataType data_type,
                                 const std::shared_ptr<TensorStorage>& tensor_storage,
                                 const intrusive::shared_ptr<LocalDepObject>& dep_object)
    : BlobObject(mem_case, shape, data_type),
      tensor_storage_(tensor_storage),
      is_shape_synced_(true),
      compute_local_dep_object_(dep_object) {
  CHECK(static_cast<bool>(shape));
  CHECK(static_cast<bool>(tensor_storage));
}

Maybe<void> EagerBlobObject::TryInitBlob() {
  if (!blob_) { JUST(InitBlob()); }
  return Maybe<void>::Ok();
}

Maybe<void> EagerBlobObject::InitBlob() { return InitBlobWithOffset(0); }

Maybe<void> EagerBlobObject::InitBlobWithOffset(const int64_t offset) {
  CHECK_NE_OR_RETURN(blob_desc_.data_type(), DataType::kInvalidDataType);
  if (!blob_desc_.shape().is_initialized()) { blob_desc_.set_shape(Shape(DimVector{})); }
  {
    header_buffer_.reset();
    int64_t header_byte_size = blob_desc_.AlignedByteSizeOfBlobHeader();
    header_buffer_ = std::make_unique<char[]>(header_byte_size);
  }
  blob_.reset(new Blob(*mem_case_, &blob_desc_, header_buffer_.get(), nullptr, offset));
  return Maybe<void>::Ok();
}

Maybe<void> EagerBlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL_OR_RETURN(allocator);
  Blob* blob = mut_blob();
  CHECK_NOTNULL_OR_RETURN(blob);
  size_t required_body_bytes = blob->AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL_OR_RETURN(tensor_storage_->blob_dptr());
    return Maybe<void>::Ok();
  }
  if (tensor_storage_->blob_dptr() != nullptr) {
    CHECK_GE_OR_RETURN(tensor_storage_->blob_bytes(), required_body_bytes);
    return Maybe<void>::Ok();
  }
  {
    // reset tensor_storage_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
      if (IsShuttingDown()) { return; }
      allocator->Deallocate(dptr, required_body_bytes);
    };
    char* dptr = nullptr;
    allocator->Allocate(&dptr, required_body_bytes);
    tensor_storage_->set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>(dptr, Free),
                                   required_body_bytes);

    blob->reset_dptr(dptr);
    InitNonPODTypeBlobIfNeed(tensor_storage_->non_pod_allocator(), blob_.get());
  }
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
