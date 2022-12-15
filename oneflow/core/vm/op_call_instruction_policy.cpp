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

#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/framework/stream_get_stream_type_name.h"
#include "oneflow/core/vm/stream_get_allocator_stream_type.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

struct OpCallInstructionUtil final {
  static inline Maybe<void> Prepare(OpCallInstructionPolicy* op_call_instruction_policy,
                                    Instruction* instruction) {
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      InferTempStorageSize(op_call_instruction_policy);
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> Compute(OpCallInstructionPolicy* op_call_instruction_policy,
                                    Instruction* instruction) {
    Allocator* allocator = instruction->mut_stream()->mut_stream_policy()->mut_allocator();
    JUST(AllocateOutputBlobsMemory(op_call_instruction_policy, allocator, instruction));
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      JUST(TryAllocateTempStorage(op_call_instruction_policy, allocator));
    }
    ep::Stream* stream = instruction->mut_stream()->mut_stream_policy()->stream();
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (op_call_instruction_policy->user_opkernel()->has_state_or_cache()) {
      TryInitOpKernelStateAndCache(op_call_instruction_policy, stream, &state, &cache);
    }
    OpKernelCompute(op_call_instruction_policy, stream, state, cache);
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      DeallocateTempStorage(op_call_instruction_policy, allocator);
    }
    return Maybe<void>::Ok();
  }

 private:
  static inline void InferTempStorageSize(OpCallInstructionPolicy* op_call_instruction_policy) {
    auto* tmp_tensor = op_call_instruction_policy->mut_call_ctx()->mut_tmp_tensor();
    size_t temp_size = op_call_instruction_policy->opkernel().InferTmpSize(
        op_call_instruction_policy->mut_call_ctx(), op_call_instruction_policy->user_opkernel());
    tmp_tensor->set_tmp_buffer_size(temp_size);
  }

  static inline void TryInitOpKernelStateAndCache(
      OpCallInstructionPolicy* op_call_instruction_policy, ep::Stream* stream,
      user_op::OpKernelState** state, user_op::OpKernelCache** cache) {
    OF_PROFILER_RANGE_GUARD("TryInitOpKernelStateAndCache");
    if (likely(op_call_instruction_policy->op_interp_ctx().state)) {
      *state = op_call_instruction_policy->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    op_call_instruction_policy->mut_opkernel()->TryInitOpKernelStateAndCache(
        op_call_instruction_policy->mut_call_ctx(), stream,
        op_call_instruction_policy->user_opkernel(), state, cache);
  }

  // Returns true if allocation happened.
  static inline Maybe<void> AllocateOutputBlobsMemory(
      OpCallInstructionPolicy* op_call_instruction_policy, Allocator* allocator,
      Instruction* instruction) {
    OF_PROFILER_RANGE_GUARD("AllocateOutputBlobsMemory");
    StreamType stream_type = instruction->stream().stream_type();
    StreamType allocator_stream_type = JUST(GetAllocatorStreamType::Visit(stream_type));
    for (const auto& blob_object : op_call_instruction_policy->outputs()) {
      if (JUST(blob_object->TryAllocateBlobBodyMemory(allocator))) {
        CHECK_OR_RETURN(stream_type == allocator_stream_type)
            << "no allocator supported on stream type " << GetStreamTypeName::Visit(stream_type);
      }
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorage(
      OpCallInstructionPolicy* op_call_instruction_policy, Allocator* allocator) {
    OF_PROFILER_RANGE_GUARD("TryAllocateTempStorage");
    auto* tmp_tensor = op_call_instruction_policy->mut_call_ctx()->mut_tmp_tensor();
    size_t byte_size = tmp_tensor->tmp_buffer_size();
    if (byte_size > 0) {
      char* mem_ptr = nullptr;
      JUST(allocator->Allocate(&mem_ptr, byte_size));
      tmp_tensor->set_tmp_buffer_ptr(mem_ptr);
    }
    return Maybe<void>::Ok();
  }

  static inline void DeallocateTempStorage(OpCallInstructionPolicy* op_call_instruction_policy,
                                           Allocator* allocator) {
    auto* tmp_tensor = op_call_instruction_policy->mut_call_ctx()->mut_tmp_tensor();
    allocator->Deallocate(tmp_tensor->mut_tmp_buffer_ptr(), tmp_tensor->tmp_buffer_size());
  }

  static inline void OpKernelCompute(OpCallInstructionPolicy* op_call_instruction_policy,
                                     ep::Stream* stream, user_op::OpKernelState* state,
                                     user_op::OpKernelCache* cache) {
    auto* user_kernel = op_call_instruction_policy->user_opkernel();
    op_call_instruction_policy->mut_opkernel()->Compute(op_call_instruction_policy->mut_call_ctx(),
                                                        stream, user_kernel, state, cache);
  }
};

OpCallInstructionPolicy::OpCallInstructionPolicy(
    Stream* vm_stream, const std::shared_ptr<one::StatefulOpKernel>& opkernel,
    EagerBlobObjectList&& inputs, EagerBlobObjectList&& outputs,
    const std::shared_ptr<const one::GlobalTensorInferResult>& global_tensor_infer_result,
    const one::OpExprInterpContext& op_interp_ctx,
    const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode)
    : vm_stream_(vm_stream),
      call_ctx_(ComposedAttrMap(op_interp_ctx.attrs, opkernel->base_attrs()), std::move(inputs),
                std::move(outputs), global_tensor_infer_result, op_interp_ctx,
                opkernel->mem_case()),
      opkernel_(opkernel),
      user_opkernel_(nullptr),
      infer_tmp_size_fn_(nullptr),
      need_temp_storage_(false),
      dev_vm_dep_object_consume_mode_(dev_vm_dep_object_consume_mode),
      input_dependences_(),
      output_dependences_() {
  ForEachConstDependence([&](auto* dep) { input_dependences_.emplace_back(dep); });
  ForEachMutDependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
  ForEachMut2Dependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
  InitStreamSequentialDependence();
}

Maybe<void> OpCallInstructionPolicy::Init() {
  return mut_opkernel()->ChooseOpKernel(&call_ctx_, &user_opkernel_, &need_temp_storage_);
}

template<typename DoEachT>
void OpCallInstructionPolicy::ForEachConstDependence(const DoEachT& DoEach) const {
  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4const_ibns()) {
    const auto& input = input_list.at(index);
    DoEach(CHECK_JUST(input->compute_local_dep_object()));
  }
}

void OpCallInstructionPolicy::InitStreamSequentialDependence() {
  auto* device_schedule_dep_object = vm_stream_->schedule_local_dep_object().get();
  if (IsCommNetStream::Visit(vm_stream_->stream_type())) {
    // Sequantialize nccl instructions to avoid deadlock
    stream_sequential_dependence_ = device_schedule_dep_object;
  } else {
    // Sequantialize instructions to avoid explosive memory allocation of source ops
    if (dev_vm_dep_object_consume_mode() == one::DevVmDepObjectConsumeMode::MUTABLE) {
      stream_sequential_dependence_ = device_schedule_dep_object;
    } else if (opkernel().input_tuple_indexes4const_ibns().empty()
               && opkernel().input_tuple_indexes4mut_ibns().empty()) {
      stream_sequential_dependence_ = device_schedule_dep_object;
    }
  }
}

template<typename DoEachT>
void OpCallInstructionPolicy::ForEachMutDependence(const DoEachT& DoEach) const {
  for (const auto& transport_dependence : vm_stream_->transport_dependences()) {
    DoEach(transport_dependence.get());
  }

  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4mut_ibns()) {
    const auto& input = input_list.at(index);
    DoEach(CHECK_JUST(input->compute_local_dep_object()));
  }
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut_obns()) {
    const auto& output = output_list.at(index);
    DoEach(CHECK_JUST(output->compute_local_dep_object()));
  }
}

template<typename DoEachT>
void OpCallInstructionPolicy::ForEachMut2Dependence(const DoEachT& DoEach) const {
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut2_obns()) {
    const auto& output = output_list.at(index);
    DoEach(CHECK_JUST(output->compute_local_dep_object()));
  }
}

Maybe<void> OpCallInstructionPolicy::Prepare(vm::Instruction* instruction) {
  return OpCallInstructionUtil::Prepare(this, instruction);
}

void OpCallInstructionPolicy::Compute(vm::Instruction* instruction) {
  CHECK_JUST_MSG(OpCallInstructionUtil::Compute(this, instruction), instruction->DebugName());
}

std::string OpCallInstructionPolicy::DebugName(const vm::Instruction& instruction) const {
  return opkernel().op_type_name() + ":OpCall";
}

}  // namespace vm
}  // namespace oneflow
