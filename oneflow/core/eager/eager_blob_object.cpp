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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {
namespace vm {

namespace {
Maybe<VmLocalDepObject> GetVmLocalDepObject(
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  return parallel_desc != nullptr
             ? Maybe<VmLocalDepObject>(std::make_shared<VmLocalDepObject>(parallel_desc))
             : Error::Unimplemented();
}
}  // namespace

EagerBlobObject::EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                                 const std::shared_ptr<Shape>& shape, DataType data_type,
                                 const std::shared_ptr<TensorBuffer>& tensor_buffer,
                                 const std::shared_ptr<const ParallelDesc>& parallel_desc)
    : BlobObject(mem_case, shape, data_type),
      tensor_buffer_(tensor_buffer),
      blob_body_bytes_(0),
      is_shape_synced_(true),
      compute_local_dep_object_(GetVmLocalDepObject(parallel_desc)) {
  CHECK(static_cast<bool>(shape));
  CHECK(static_cast<bool>(tensor_buffer));
  non_pod_initer_ = std::make_unique<MemoryAllocator>();
}

Maybe<void> EagerBlobObject::TryInitBlob() {
  if (!blob_) { JUST(InitBlob()); }
  return Maybe<void>::Ok();
}

Maybe<void> EagerBlobObject::InitBlob() {
  CHECK_NE_OR_RETURN(blob_desc_.data_type(), DataType::kInvalidDataType);
  if (!blob_desc_.shape().is_initialized()) { blob_desc_.set_shape(Shape(DimVector{})); }
  {
    header_buffer_.reset();
    int64_t header_byte_size = blob_desc_.AlignedByteSizeOfBlobHeader();
    const auto& FreeHeader = [header_byte_size](char* dptr) { std::free(dptr); };
    char* ptr = reinterpret_cast<char*>(std::malloc(header_byte_size));
    header_buffer_ = std::unique_ptr<char, std::function<void(char*)>>(ptr, FreeHeader);
  }
  blob_.reset(new Blob(*mem_case_, &blob_desc_, header_buffer_.get(), nullptr));
  return Maybe<void>::Ok();
}

Maybe<void> EagerBlobObject::TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) {
  vm::Allocator* allocator = device_ctx->mut_allocator();
  CHECK_NOTNULL_OR_RETURN(allocator);
  Blob* blob = mut_blob();
  CHECK_NOTNULL_OR_RETURN(blob);
  const std::size_t required_body_bytes = blob->AlignedByteSizeOfBlobBody();
  if (required_body_bytes == 0) {
    CHECK_ISNULL_OR_RETURN(blob->dptr());
    return Maybe<void>::Ok();
  }
  if (blob->dptr() != nullptr) {
    CHECK_EQ_OR_RETURN(blob_body_bytes_, required_body_bytes);
    return Maybe<void>::Ok();
  }
  {
    // reset tensor_buffer_;
    const auto& Free = [allocator, required_body_bytes](char* dptr) {
      if (IsShuttingDown()) { return; }
      allocator->Deallocate(dptr, required_body_bytes);
    };
    char* dptr = nullptr;
    allocator->Allocate(&dptr, required_body_bytes);
    tensor_buffer_->set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>(dptr, Free));
    blob->reset_dptr(dptr);
    InitNonPODTypeBlobIfNeed(non_pod_initer_.get(), blob_.get());
  }
  blob_body_bytes_ = required_body_bytes;
  return Maybe<void>::Ok();
}

Maybe<void> DTREagerBlobObject::InitBlobAttrs(std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  // reset DTREageBlobObject properties
  compute_time_ = 0;
  pinned_ = 0;

  // current time 
  update_access_time();
  // last_access_time_ = Global<one::DTRTensorPool>::Get()->duration();

  compute_op_ = operand;

  return Maybe<void>::Ok();
}

void DTREagerBlobObject::update_access_time() {
  last_access_time_ = Global<one::DTRTensorPool>::Get()->duration();
}

void DTREagerBlobObject::update_user_ops(std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  user_ops_.emplace_back(operand);
}

bool DTREagerBlobObject::is_in_memory() {
  return !evict_flag_;
  // return (tensor_buffer_.get()->blob_dptr() != nullptr);
}

Maybe<double> DTREagerBlobObject::parent_cost() {
  double cost = 0;

  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(compute_op_.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : *ptr->inputs()) {
    CHECK_OR_RETURN(static_cast<bool>(input.get()));
    auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    if (!dtr_blob_object->is_in_memory()) {
      auto com_time = dtr_blob_object->compute_time();
      auto p_cost = JUST(dtr_blob_object->parent_cost());
      cost = cost + com_time + p_cost;
    }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::child_cost() {
  double cost = 0;

  for (auto operand: user_ops_) {
    auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(ptr);
    for (const auto& input : *ptr->outputs()) {
      CHECK_OR_RETURN(static_cast<bool>(input.get()));
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) {
        auto com_time = dtr_blob_object->compute_time();
        auto c_cost = JUST(dtr_blob_object->child_cost());
        cost = cost + com_time + c_cost;
      }
    }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::neighbor_cost() {
  auto p_cost = JUST(parent_cost());
  auto c_cost = JUST(child_cost());
  return p_cost + c_cost + compute_time_;
}

Maybe<double> DTREagerBlobObject::cost() {
  auto n_cost = JUST(neighbor_cost());
  return n_cost / blob_body_bytes_ / last_access_time_; 
}

}  // namespace vm
}  // namespace oneflow
