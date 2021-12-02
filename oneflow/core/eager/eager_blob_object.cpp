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
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace vm {

EagerBlobObject::EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case,
                                 const std::shared_ptr<Shape>& shape, DataType data_type,
                                 const std::shared_ptr<TensorBuffer>& tensor_buffer,
                                 const Optional<LocalDepObject*>& dep_object)
    : BlobObject(mem_case, shape, data_type),
      tensor_buffer_(tensor_buffer),
      blob_body_bytes_(0),
      is_shape_synced_(true),
      compute_local_dep_object_(dep_object) {
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
    header_buffer_ = std::make_unique<char[]>(header_byte_size);
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

DTREagerBlobObject::~DTREagerBlobObject() {
  clear_invalid_object();
  non_pod_initer_.reset();
  tensor_buffer_.reset();
  blob_.reset();
}

void DTREagerBlobObject::clear_invalid_object() {
  if (IsShuttingDown()) { return; }
  CHECK_JUST(Global<one::DTRTensorPool>::Get()->clear());
}

Maybe<void> DTREagerBlobObject::InitBlobAttrs(
    std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand) {
  // reset DTREageBlobObject properties
  compute_time_ = 0;
  pinned_ = 0;

  // current time
  update_access_time();
  // last_access_time_ = Global<one::DTRTensorPool>::Get()->duration();
  // TODO: unique_ptr
  compute_op_ = std::make_unique<DTRInstrOperand>(
      operand->shared_opkernel(), operand->inputs(), operand->outputs(),
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode());
  // compute_op_ = operand;
  could_evict_ = (input_size() > 0) && could_evict_;

  node = std::make_shared<DisjNode>(0);

  return Maybe<void>::Ok();
}

void DTREagerBlobObject::update_access_time() {
  last_access_time_ = Global<one::DTRTensorPool>::Get()->duration();
}

void DTREagerBlobObject::update_user_ops(
    std::shared_ptr<vm::LocalCallOpKernelPhyInstrOperand>& operand) {
  // TODO unique_ptr
  user_ops_.emplace_back(std::make_unique<DTRInstrOperand>(
      operand->shared_opkernel(), operand->inputs(), operand->outputs(),
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode()));
}

bool DTREagerBlobObject::is_in_memory() const {
  // return !evict_flag_;
  return (tensor_buffer_->blob_dptr() != nullptr);
}

Maybe<double> DTREagerBlobObject::parent_cost() const {
  double cost = 0;

  auto* ptr = dynamic_cast<DTRInstrOperand*>(compute_op_.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : ptr->inputs()) {
    if (!input.expired()) {
      auto object = input.lock();
      const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) {
        auto com_time = dtr_blob_object->compute_time();
        auto p_cost = JUST(dtr_blob_object->parent_cost());
        cost = cost + com_time + p_cost;
      }
    }

    // CHECK_OR_RETURN(static_cast<bool>(input.get()));
    // auto object = input.get();
    // const auto* dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    // // CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    // if (dtr_blob_object == nullptr) {
    //   continue;
    // } else {
    //   if (!dtr_blob_object->is_in_memory()) {
    //     auto com_time = dtr_blob_object->compute_time();
    //     auto p_cost = JUST(dtr_blob_object->parent_cost());
    //     cost = cost + com_time + p_cost;
    //   }
    // }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::child_cost() const {
  double cost = 0;

  for (int i = 0; i < user_ops_.size(); ++i) {
    const auto* ptr = dynamic_cast<DTRInstrOperand*>(JUST(user_op(i)));
    CHECK_NOTNULL_OR_RETURN(ptr);
    for (const auto& output : ptr->outputs()) {
      if (!output.expired()) {
        auto object = output.lock();
        const auto dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(object);
        CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
        if (!dtr_blob_object->is_in_memory()) {
          auto com_time = dtr_blob_object->compute_time();
          auto c_cost = JUST(dtr_blob_object->child_cost());
          cost = cost + com_time + c_cost;
        }
      }

      // CHECK_OR_RETURN(static_cast<bool>(output.get()));
      // auto object = output.get();
      // CHECK_NOTNULL_OR_RETURN(object);
      // const auto* dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(output.get());
      // // CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      // if (dtr_blob_object == nullptr) {
      //   continue;
      // } else {
      //   if (!dtr_blob_object->is_in_memory()) {
      //     auto com_time = dtr_blob_object->compute_time();
      //     auto c_cost = JUST(dtr_blob_object->child_cost());
      //     cost = cost + com_time + c_cost;
      //   }
      // }
    }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::neighbor_cost() const {
  auto p_cost = JUST(parent_cost());
  auto c_cost = JUST(child_cost());
  return p_cost + c_cost + compute_time_;
}

Maybe<double> DTREagerBlobObject::approx_neighbor_cost() const {
  double cost = 0;
  const auto& inputs = compute_op_->inputs();
  for (int i = 0; i < inputs.size(); ++i) {
    if (auto tmp = inputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      double p_cost = Global<one::DTRTensorPool>::Get()->find_father(dtr_blob_object->node)->compute_time();
      cost += p_cost;
    }
  }

  const auto& outputs = compute_op_->outputs();
  for (int i = 0; i < outputs.size(); ++i) {
    if (auto tmp = outputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      double c_cost = Global<one::DTRTensorPool>::Get()->find_father(dtr_blob_object->node)->compute_time();
      cost += c_cost;
    }
  }

  return cost;
}

Maybe<double> DTREagerBlobObject::cost() const {
  auto n_cost = JUST(approx_neighbor_cost());
  // auto n_cost = JUST(neighbor_cost());
  double time_since_last_access = Global<one::DTRTensorPool>::Get()->duration() - last_access_time_;
  if (oneflow::DTRDebugEnabled()) {
    std::cout << "n_cost " << n_cost << ", blob_body_bytes_ " << blob_body_bytes_ << ", time_since_last_access " << time_since_last_access << std::endl;
  }
  return n_cost / blob_body_bytes_ / time_since_last_access;
}

size_t DTREagerBlobObject::input_size() const {
  const auto& ptr = dynamic_cast<DTRInstrOperand*>(compute_op_.get());
  return ptr->inputs().size();
}

const std::string& DTREagerBlobObject::compute_op_type_name() const {
  return compute_op_->shared_opkernel()->op_type_name();
}

bool DTREagerBlobObject::is_evictable() const {
  if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "nll") { return false; }
  // if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "conv_filter_grad") { return false; }
  // if (compute_op_->shared_opkernel()->user_op_conf_->op_type_name() == "matmul") { return false; }
  return could_evict_;
}

}  // namespace vm
}  // namespace oneflow
