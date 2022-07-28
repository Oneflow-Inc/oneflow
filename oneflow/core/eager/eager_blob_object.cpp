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
      mem_ptr_for_allocation_compuation_pipelining_(nullptr),
      inited_mem_ptr_for_allocation_compuation_pipelining_(false),
      is_non_pod_object_placement_newed_(false),
      pin_memory_(false),
      compute_local_dep_object_(dep_object),
      blob_desc_(static_cast<bool>(dynamic_local_tensor_meta)
                     ? std::const_pointer_cast<Shape>(dynamic_local_tensor_meta->shape_ptr())
                     : std::const_pointer_cast<Shape>(static_local_tensor_meta->shape_ptr()),
                 static_cast<bool>(dynamic_local_tensor_meta)
                     ? std::const_pointer_cast<Stride>(dynamic_local_tensor_meta->stride_ptr())
                     : std::const_pointer_cast<Stride>(static_local_tensor_meta->stride_ptr()),
                 data_type),
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
Shape* EagerBlobObject::mut_shape() {
  CHECK(dynamic_local_tensor_meta_);
  return std::const_pointer_cast<one::MutLocalTensorMeta>(dynamic_local_tensor_meta_)->mut_shape();
}
const Stride& EagerBlobObject::stride() const {
  if (dynamic_local_tensor_meta_) {
    return dynamic_local_tensor_meta_->stride();
  } else {
    return static_local_tensor_meta_->stride();
  }
}
Stride* EagerBlobObject::mut_stride() {
  CHECK(dynamic_local_tensor_meta_);
  return std::const_pointer_cast<one::MutLocalTensorMeta>(dynamic_local_tensor_meta_)->mut_stride();
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

Blob* EagerBlobObject::blob() {
  if (!blob_) {
    blob_.reset(new Blob(*mem_case_, &blob_desc_, mut_header_ptr(), mut_dptr<char>()));
  }
  return blob_.get();
}

void EagerBlobObject::set_storage_offset(const int64_t offset) { storage_offset_ = offset; }

void EagerBlobObject::TryInitNonPODTypeEagerBlobObjectIfNeed() {
  if (!IsPODDataType(data_type())) {
    if (!is_non_pod_object_placement_newed_) {
      InitNonPODTypeEagerBlobObjectIfNeed(tensor_storage_->non_pod_allocator(), this);
      is_non_pod_object_placement_newed_ = true;
    }
  }
}

Maybe<void> EagerBlobObject::TryAllocateBlobBodyMemory(vm::Allocator* allocator) {
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
    InitMemPtrForAllocationComputationPipelining();
  }
  InitOrCheckMemPtrForAllocationComputationPipelining();
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
