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
#include <fmt/format.h>
#include <algorithm>
#include "oneflow/core/common/env_var/vm.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"
#include "oneflow/core/vm/dtr_disjoint_set.h"
#include "oneflow/core/vm/dtr_env.h"
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/framework/stream_get_stream_type_name.h"
#include "oneflow/core/vm/stream_get_allocator_stream_type.h"
#include "oneflow/core/profiler/profiler.h"
#include "fmt/core.h"

namespace oneflow {
namespace vm {

Maybe<void> _IncReferenceNumOfRecomputedTensor(
    const OpCallInstructionPolicy& op_call_instruction_policy, int& pinned_num,
    std::set<const DtrOpCallInstructionPolicy*>& visited_ops) {
  VLOG(1) << "op is " << op_call_instruction_policy.opkernel().op_type_name();
  for (int i = 0; i < op_call_instruction_policy.inputs().size(); i++) {
    const auto& input = op_call_instruction_policy.inputs().at(i);
    input->tensor_storage()->Pin();
    VLOG(1) << "No." << i << " input is in memory? " << input->tensor_storage()->is_in_memory();
    if (!input->tensor_storage()->is_in_memory()) {
      OpCallInstructionPolicy tmp_op = input->tensor_storage()->compute_op();
      if (!input->tensor_storage()->is_needed_by_backward()) {
        Singleton<dtr::Env>::Get()->need_eager_eviction_storages.insert(
            input->tensor_storage().get());
      }

      if (visited_ops.find(input->tensor_storage()->dtr_compute_op().get()) == visited_ops.end()) {
        visited_ops.insert(input->tensor_storage()->dtr_compute_op().get());
        JUST(_IncReferenceNumOfRecomputedTensor(tmp_op, pinned_num, visited_ops));
      }
    } else {
      pinned_num++;
    }
  }
  VLOG(1) << "op " << op_call_instruction_policy.opkernel().op_type_name() << " end";
  return Maybe<void>::Ok();
}

Maybe<int> IncReferenceNumOfRecomputedTensor(
    const OpCallInstructionPolicy& op_call_instruction_policy) {
  int pinned_num = 0;
  std::set<const DtrOpCallInstructionPolicy*> visited_ops;
  JUST(_IncReferenceNumOfRecomputedTensor(op_call_instruction_policy, pinned_num, visited_ops));
  return pinned_num;
}

struct OpCallInstructionUtil final {
  static inline Maybe<void> Prepare(OpCallInstructionPolicy* op_call_instruction_policy,
                                    Instruction* instruction) {
    VLOG(1) << "prepare " << op_call_instruction_policy->opkernel().op_type_name() << std::endl;
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      InferTempStorageSize(op_call_instruction_policy);
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> RematInputs(OpCallInstructionPolicy* op_call_instruction_policy,
                                        vm::Stream* vm_stream, bool first, bool recompute) {
    CHECK_OR_RETURN(!ThreadLocalEnvBool<ONEFLOW_VM_MULTI_THREAD>());
    Singleton<dtr::Env>::Get()->current_op_type_name =
        op_call_instruction_policy->opkernel().op_type_name();
    VLOG(2) << "set current op type name to " << Singleton<dtr::Env>::Get()->current_op_type_name
            << std::endl;
    if (first) { JUST(IncReferenceNumOfRecomputedTensor(*op_call_instruction_policy)); }
    VLOG(1) << "compute " << op_call_instruction_policy->opkernel().op_type_name() << std::endl;
    VLOG(1) << "input num " << op_call_instruction_policy->inputs().size() << std::endl;

    for (int i = 0; i < op_call_instruction_policy->inputs().size(); i++) {
      const auto& x = op_call_instruction_policy->inputs().at(i);
      if (!x->tensor_storage()->is_in_memory()) {
        VLOG(1) << "recompute No." << i << " input by "
                << x->tensor_storage()->compute_op_type_name()
                << ". Storage id: " << x->tensor_storage()->id();
        OpCallInstructionPolicy tmp_op = x->tensor_storage()->compute_op();
        JUST(Compute(&tmp_op, vm_stream, false, true));
        // JUST(vm_stream->mut_stream_policy()->stream()->Sync());
        // CHECK_OR_RETURN(x->tensor_storage()->is_in_memory());
      }
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> EagerEvictRemattedTensors(
      OpCallInstructionPolicy* op_call_instruction_policy, vm::Stream* vm_stream, bool first,
      bool recompute) {
    auto& need_eager_eviction_storages = Singleton<dtr::Env>::Get()->need_eager_eviction_storages;
    for (const auto& x : op_call_instruction_policy->inputs()) {
      x->tensor_storage()->Unpin();
      if (x->tensor_storage()->num_pinned() == 0
          && need_eager_eviction_storages.count(x->tensor_storage().get()) > 0) {
        need_eager_eviction_storages.erase(x->tensor_storage().get());
        x->tensor_storage()->Evict(true);
      }
    }
    if (first) {
      if (!need_eager_eviction_storages.empty()) {
        for (const auto& storage : need_eager_eviction_storages) {
          VLOG(1) << "not empty, storage id: " << storage->id();
        }
      }
      CHECK_OR_RETURN(need_eager_eviction_storages.empty());
    }
    return Maybe<void>::Ok();
  }
  static inline Maybe<void> UpdateRematInfo(OpCallInstructionPolicy* op_call_instruction_policy,
                                            vm::Stream* vm_stream, bool first, bool recompute,
                                            bool include_input, bool include_output,
                                            const std::vector<bool>& storage_is_initialized) {
    if (include_output) {
      const std::unique_ptr<OpCallInstructionPolicy> compute_op = [&]() {
        auto compute_op = std::make_unique<OpCallInstructionPolicy>(*op_call_instruction_policy);
        for (int i = 0; i < op_call_instruction_policy->outputs().size(); i++) {
          const auto& y = op_call_instruction_policy->outputs().at(i);
          if (y->tensor_storage()->is_eviction_disabled()) { continue; }
          if (storage_is_initialized[i] && !recompute) {
            VLOG(1) << "y->tensor_storage()->is_initialized(), y's op is "
                    << y->tensor_storage()->compute_op_type_name() << std::endl;
            compute_op = std::make_unique<OpCallInstructionPolicy>(
                Singleton<dtr::Env>::Get()->update_tensor_with_storage(y->tensor_storage().get(),
                                                                       op_call_instruction_policy));
          }
        }
        return compute_op;
      }();
      std::shared_ptr<DtrOpCallInstructionPolicy> dtr_compute_op =
          std::make_shared<DtrOpCallInstructionPolicy>(*compute_op);
      double compute_time = JUST(dtr::GetComputeTime(*compute_op));
      for (int i = 0; i < op_call_instruction_policy->outputs().size(); i++) {
        const auto& y = op_call_instruction_policy->outputs().at(i);
        y->tensor_storage()->Pin();
        if (!recompute && !y->tensor_storage()->is_eviction_disabled()) {
          y->tensor_storage()->set_compute_op(dtr_compute_op, compute_time);
        }
        y->tensor_storage()->Unpin();
        y->tensor_storage()->Access();
        dtr::DisjointSet::update_after_compute(y->tensor_storage().get());
      }
    }
    if (include_input) {
      for (int i : op_call_instruction_policy->opkernel().input_tuple_indexes4mut_ibns()) {
        const auto& x = op_call_instruction_policy->inputs().at(i);
        x->tensor_storage()->set_eviction_disabled(true);
      }

      for (int i = 0; i < op_call_instruction_policy->inputs().size(); i++) {
        const auto& x = op_call_instruction_policy->inputs().at(i);
        x->tensor_storage()->Access();
      }
    }

    if (recompute) { Singleton<dtr::Env>::Get()->add_recomputation_num(); }
    Singleton<dtr::Env>::Get()->add_time(JUST(dtr::GetComputeTime(*op_call_instruction_policy)));
    VLOG(1) << "end compute " << op_call_instruction_policy->opkernel().op_type_name() << std::endl;
    Singleton<dtr::Env>::Get()->current_op_type_name = "None";
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> Compute(OpCallInstructionPolicy* op_call_instruction_policy,
                                    vm::Stream* vm_stream, bool first, bool recompute) {
    Allocator* allocator = vm_stream->mut_stream_policy()->mut_allocator();
    bool remat_input = vm_stream->device()->with_remat();
    bool remat_output = vm_stream->device()->with_remat();
    if (!remat_input && remat_output) {
      for (const auto& y : op_call_instruction_policy->outputs()) {
        y->tensor_storage()->set_eviction_disabled(true);
      }
    }
    std::cout << "op: " << op_call_instruction_policy->opkernel().op_type_name() << std::endl;
    std::cout << "remat_input: " << remat_input << ", remat_output: " << remat_output << std::endl;
    if (remat_input) { JUST(RematInputs(op_call_instruction_policy, vm_stream, first, recompute)); }
    std::vector<bool> storage_is_initialized;
    if (remat_output) {
      storage_is_initialized.reserve(op_call_instruction_policy->outputs().size());
      for (const auto& y : op_call_instruction_policy->outputs()) {
        storage_is_initialized.push_back(y->tensor_storage()->is_initialized());
      }
    }
    JUST(AllocateOutputBlobsMemory(op_call_instruction_policy, allocator, vm_stream));
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      JUST(TryAllocateTempStorage(op_call_instruction_policy, allocator));
    }
    ep::Stream* stream = vm_stream->mut_stream_policy()->stream();
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (op_call_instruction_policy->user_opkernel()->has_state_or_cache()) {
      TryInitOpKernelStateAndCache(op_call_instruction_policy, stream, &state, &cache);
    }
    OpKernelCompute(op_call_instruction_policy, stream, state, cache);
    if (unlikely(op_call_instruction_policy->need_temp_storage())) {
      DeallocateTempStorage(op_call_instruction_policy, allocator);
    }
    if (remat_input) {
      JUST(EagerEvictRemattedTensors(op_call_instruction_policy, vm_stream, first, recompute));
    }
    JUST(UpdateRematInfo(op_call_instruction_policy, vm_stream, first, recompute, remat_input,
                         remat_output, storage_is_initialized));
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
      const vm::Stream* vm_stream) {
    OF_PROFILER_RANGE_GUARD("AllocateOutputBlobsMemory");
    StreamType stream_type = vm_stream->stream_type();
    StreamType allocator_stream_type = JUST(GetAllocatorStreamType::Visit(stream_type));
    for (const auto& blob_object : op_call_instruction_policy->outputs()) {
      if (JUST(blob_object->TryAllocateBlobBodyMemory(allocator))) {
        CHECK_OR_RETURN(stream_type == allocator_stream_type)
            << "no allocator supported on stream type " << GetStreamTypeName::Visit(stream_type);
        if (auto* dtr_allocator = dynamic_cast<vm::DtrEpAllocatorProxy*>(allocator)) {
          dtr_allocator->allocator->Mark(blob_object.get(),
                                         static_cast<const char*>(blob_object->dptr()));
        }
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
    : id(-1),
      vm_stream_(vm_stream),
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
  id = unique_id();
  return mut_opkernel()->ChooseOpKernel(&call_ctx_, &user_opkernel_, &need_temp_storage_);
}

OpCallInstructionPolicy::OpCallInstructionPolicy(const DtrOpCallInstructionPolicy& policy)
    : id(unique_id()),
      vm_stream_(policy.vm_stream_),
      call_ctx_(policy.dtr_call_ctx_),
      opkernel_(policy.opkernel_),
      user_opkernel_(policy.user_opkernel_),
      infer_tmp_size_fn_(policy.infer_tmp_size_fn_),
      need_temp_storage_(policy.need_temp_storage_),
      dev_vm_dep_object_consume_mode_(policy.dev_vm_dep_object_consume_mode_),
      input_dependences_(policy.input_dependences_),
      output_dependences_(policy.output_dependences_) {}

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
  CHECK_JUST_MSG(OpCallInstructionUtil::Compute(this, instruction->mut_stream(), true, false),
                 instruction->DebugName());
}

std::string OpCallInstructionPolicy::DebugName(const vm::Instruction& instruction) const {
  return opkernel().op_type_name() + ":OpCall";
}

Maybe<void> Recompute(OpCallInstructionPolicy* op_call_instruction_policy, vm::Stream* vm_stream) {
  VLOG(1) << "recompute " << op_call_instruction_policy->opkernel().op_type_name() << " manually";
  return OpCallInstructionUtil::Compute(op_call_instruction_policy, vm_stream, true, true);
}

}  // namespace vm
}  // namespace oneflow
