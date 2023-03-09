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
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/tensor_meta.h"

namespace oneflow {

namespace vm {

EagerBlobObject::EagerBlobObject(
    const std::shared_ptr<MemoryCase>& mem_case,
    const Symbol<one::LocalTensorMeta>& static_local_tensor_meta,
    const std::shared_ptr<const one::MutLocalTensorMeta>& dynamic_local_tensor_meta,
    DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage,
    const intrusive::shared_ptr<LocalDepObject>& dep_object)
    : is_dynamic_(false),
      mem_case_(mem_case),
      data_type_(data_type),
      storage_offset_(0),
      tensor_storage_(tensor_storage),
      compute_local_dep_object_(dep_object),
      static_local_tensor_meta_(static_local_tensor_meta),
      dynamic_local_tensor_meta_(dynamic_local_tensor_meta) {
  CHECK(static_cast<bool>(tensor_storage));
}

// user_op::TensorDesc overrides
const Shape& EagerBlobObject::shape() const {
  if (dynamic_local_tensor_meta_) {
    return dynamic_local_tensor_meta_->shape();
  } else {
    return static_local_tensor_meta_->shape();
  }
}
const Stride& EagerBlobObject::stride() const {
  if (dynamic_local_tensor_meta_) {
    return dynamic_local_tensor_meta_->stride();
  } else {
    return static_local_tensor_meta_->stride();
  }
}

void EagerBlobObject::set_shape(const Shape& shape) {
  CHECK(dynamic_local_tensor_meta_);
  std::const_pointer_cast<one::MutLocalTensorMeta>(dynamic_local_tensor_meta_)->set_shape(shape);
}
void EagerBlobObject::set_stride(const Stride& stride) {
  CHECK(dynamic_local_tensor_meta_);
  std::const_pointer_cast<one::MutLocalTensorMeta>(dynamic_local_tensor_meta_)->set_stride(stride);
}

MutShapeView EagerBlobObject::mut_shape_view() {
  CHECK(dynamic_local_tensor_meta_);
  return *const_cast<Shape*>(dynamic_local_tensor_meta_->shape_ptr().get());
}

std::shared_ptr<const Shape> EagerBlobObject::shape_ptr() const {
  if (dynamic_local_tensor_meta_) {
    return dynamic_local_tensor_meta_->shape_ptr();
  } else {
    return static_local_tensor_meta_->shape_ptr();
  }
}
std::shared_ptr<const Stride> EagerBlobObject::stride_ptr() const {
  if (dynamic_local_tensor_meta_) {
    return dynamic_local_tensor_meta_->stride_ptr();
  } else {
    return static_local_tensor_meta_->stride_ptr();
  }
}

void EagerBlobObject::set_storage_offset(const int64_t offset) { storage_offset_ = offset; }

Maybe<bool> EagerBlobObject::TryAllocateBlobBodyMemory(vm::Allocator* allocator) {
  size_t required_body_bytes = AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL_OR_RETURN(tensor_storage_->blob_dptr());
  } else if (tensor_storage_->blob_dptr() != nullptr) {
    CHECK_GE_OR_RETURN(tensor_storage_->blob_bytes(), ByteSizeOfBlobBody())
        << "This blob has been allocated memory, but less than needed space.";
  } else {
    char* dptr = nullptr;
    JUST(allocator->Allocate(&dptr, required_body_bytes));
    // reset tensor_storage_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
      if (IsShuttingDown()) { return; }
      allocator->Deallocate(dptr, required_body_bytes);
    };
    tensor_storage_->set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>(dptr, Free),
                                   required_body_bytes);
    InitNonPODTypeEagerBlobObjectIfNeed(tensor_storage_->non_pod_allocator(), this);
    return true;
  }
  return false;
}

const void* EagerBlobObject::raw_dptr() const {
  char* ptr = tensor_storage_->blob_dptr();
  if (tensor_storage_->blob_bytes() > 0) { CHECK_NOTNULL(ptr); }
  return ptr + storage_offset_ * GetSizeOfDataType(data_type_);
}

Maybe<void> EagerBlobObject::DeallocateBlobDataPtr() {
  tensor_storage_->Release();
  return Maybe<void>::Ok();
}

void EagerBlobObject::RegisterStorageDeleteHook(const std::function<void()>& hook) {
  tensor_storage_->RegisterStorageDeleteHook(hook);
}

const Optional<Symbol<::oneflow::Stream>>& EagerBlobObject::producer_stream() const {
  return tensor_storage_->producer_stream();
}

Maybe<void> EagerBlobObject::init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
  return tensor_storage_->init_producer_stream(producer_stream);
}

const Optional<Symbol<::oneflow::Stream>>& EagerBlobObject::last_used_stream() const {
  return tensor_storage_->last_used_stream();
}

void EagerBlobObject::set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) {
  tensor_storage_->set_last_used_stream(last_used_stream);
}

}  // namespace vm
}  // namespace oneflow
