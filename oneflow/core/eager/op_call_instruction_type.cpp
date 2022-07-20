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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/eager/op_call_instruction_type.h"
#include "oneflow/core/eager/op_call_phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/event_recorder.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace vm {

struct OpCallInstructionUtil final {
  static inline Maybe<void> Prepare(vm::Instruction* instruction) {
    auto* operand = GetCallPhyInstrOperand(*instruction);
    vm::Allocator* allocator = instruction->mut_stream()->mut_stream_policy()->mut_allocator();
    JUST(AllocateOutputBlobsMemory(operand, allocator));
    if (unlikely(operand->need_temp_storage())) {
      InferTempStorageSize(operand);
      JUST(TryAllocateTempStorage(operand, allocator));
      // Since memory block is cached in allocator, it's safe to deallocate tmp buffer before
      // kernel executed.
      DeallocateTempStorage(operand, allocator);
    }
    return Maybe<void>::Ok();
  }

  static inline void Compute(vm::Instruction* instruction) {
    auto* operand = GetCallPhyInstrOperand(*instruction);
    ep::Stream* stream = instruction->mut_stream()->mut_stream_policy()->stream();
    if (!operand->is_all_outputs_pod()) {
      for (const auto& blob_object : operand->outputs()) {
        blob_object->TryInitNonPODTypeEagerBlobObjectIfNeed();
      }
    }
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (operand->user_opkernel()->has_state_or_cache()) {
      TryInitOpKernelStateAndCache(operand, stream, &state, &cache);
    }
    OpKernelCompute(operand, stream, state, cache);
  }

  static inline OpCallPhyInstrOperand* GetCallPhyInstrOperand(const vm::Instruction& instruction) {
    auto* operand = CHECK_NOTNULL(instruction.phy_instr_operand().get());
    return CHECK_NOTNULL(dynamic_cast<OpCallPhyInstrOperand*>(operand));
  }

 private:
  static inline void InferTempStorageSize(OpCallPhyInstrOperand* operand) {
    auto* tmp_tensor = operand->mut_call_ctx()->mut_tmp_tensor();
    size_t temp_size =
        operand->opkernel().InferTmpSize(&operand->call_ctx_, operand->user_opkernel());
    tmp_tensor->set_tmp_buffer_size(temp_size);
  }

  static inline void TryInitOpKernelStateAndCache(OpCallPhyInstrOperand* operand,
                                                  ep::Stream* stream,
                                                  user_op::OpKernelState** state,
                                                  user_op::OpKernelCache** cache) {
    OF_PROFILER_RANGE_GUARD("TryInitOpKernelStateAndCache");
    if (likely(operand->op_interp_ctx().state)) {
      *state = operand->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    operand->mut_opkernel()->TryInitOpKernelStateAndCache(&operand->call_ctx_, stream,
                                                          operand->user_opkernel(), state, cache);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(OpCallPhyInstrOperand* operand,
                                                      vm::Allocator* allocator) {
    OF_PROFILER_RANGE_GUARD("AllocateOutputBlobsMemory");
    for (const auto& blob_object : operand->outputs()) {
      JUST(blob_object->TryAllocateBlobBodyMemory(allocator));
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorage(OpCallPhyInstrOperand* operand,
                                                   vm::Allocator* allocator) {
    OF_PROFILER_RANGE_GUARD("TryAllocateTempStorage");
    auto* tmp_tensor = operand->mut_call_ctx()->mut_tmp_tensor();
    size_t byte_size = tmp_tensor->tmp_buffer_size();
    if (byte_size > 0) {
      char* mem_ptr = nullptr;
      JUST(allocator->Allocate(&mem_ptr, byte_size));
      tmp_tensor->init_tmp_buffer_ptr(mem_ptr);
    }
    return Maybe<void>::Ok();
  }

  static inline void OpKernelCompute(OpCallPhyInstrOperand* operand, ep::Stream* stream,
                                     user_op::OpKernelState* state, user_op::OpKernelCache* cache) {
    auto* call_ctx = &operand->call_ctx_;
    auto* user_kernel = operand->user_opkernel();
    operand->mut_opkernel()->Compute(call_ctx, stream, user_kernel, state, cache);
  }

  static inline void DeallocateTempStorage(OpCallPhyInstrOperand* operand,
                                           vm::Allocator* allocator) {
    OF_PROFILER_RANGE_GUARD("DeallocateTempStorage");
    auto* tmp_tensor = operand->mut_call_ctx()->mut_tmp_tensor();
    allocator->Deallocate(tmp_tensor->mut_tmp_buffer_ptr(), tmp_tensor->tmp_buffer_size());
  }
};

Maybe<void> OpCallInstructionType::Prepare(vm::Instruction* instruction) const {
  return OpCallInstructionUtil::Prepare(instruction);
}

void OpCallInstructionType::Compute(vm::Instruction* instruction) const {
  OpCallInstructionUtil::Compute(instruction);
}

std::string OpCallInstructionType::DebugName(const vm::Instruction& instruction) const {
  auto* operand = CHECK_NOTNULL(instruction.phy_instr_operand().get());
  return CHECK_NOTNULL(dynamic_cast<OpCallPhyInstrOperand*>(operand))->opkernel().op_type_name()
         + ":OpCall";
}

}  // namespace vm
}  // namespace oneflow
