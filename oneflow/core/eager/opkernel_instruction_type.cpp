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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/parallel_signature.cfg.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace vm {

struct LocalCallOpKernelUtil final {
  static inline void Compute(const vm::InstructionMsg& instr_msg) {
    OF_PROFILER_RANGE_PUSH("ResetPrior");
    auto* operand = LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instr_msg);
    DeviceCtx* device_ctx = instr_msg.phy_instr_stream()->device_ctx().get();
    operand->mut_call_ctx()->device_ctx = device_ctx;
    OF_PROFILER_RANGE_POP();
    OF_PROFILER_RANGE_PUSH("AllocateOutputBlobsMemory");
    AllocateOutputBlobsMemory(operand, device_ctx);
    OF_PROFILER_RANGE_POP();
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_PUSH("TryAllocateTempStorage");
      InferTempStorageSize(operand);
      TryAllocateTempStorage(operand, device_ctx);
      OF_PROFILER_RANGE_POP();
    }
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (operand->user_opkernel()->has_state_or_cache()) {
      OF_PROFILER_RANGE_PUSH("TryInitOpKernelStateAndCache");
      TryInitOpKernelStateAndCache(operand, device_ctx, &state, &cache);
      OF_PROFILER_RANGE_POP();
    }
    OpKernelCompute(operand, device_ctx, state, cache);
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_PUSH("DeallocateTempStorage");
      DeallocateTempStorage(operand, device_ctx);
      OF_PROFILER_RANGE_POP();
    }
    operand->mut_call_ctx()->device_ctx = nullptr;
  }

  static inline LocalCallOpKernelPhyInstrOperand* GetLocalCallOpKernelPhyInstrOperand(
      const vm::InstructionMsg& instr_msg) {
    auto* operand = CHECK_NOTNULL(instr_msg.phy_instr_operand().get());
    return CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand));
  }

 private:
  static inline void InferTempStorageSize(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& InferTmpSizeFn = operand->infer_tmp_size_fn();
    CHECK(static_cast<bool>(InferTmpSizeFn))
        << operand->opkernel().op_conf().user_conf().op_type_name();
    operand->mut_call_ctx()->tmp_buffer_size = InferTmpSizeFn(operand->opkernel().op_infer_ctx());
  }

  static inline void TryInitOpKernelStateAndCache(LocalCallOpKernelPhyInstrOperand* operand,
                                                  DeviceCtx* device_ctx,
                                                  user_op::OpKernelState** state,
                                                  user_op::OpKernelCache** cache) {
    if (likely(operand->op_interp_ctx().state)) {
      *state = operand->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    operand->mut_opkernel()->TryInitOpKernelStateAndCache(operand->user_opkernel(), state, cache);
  }

  static inline void AllocateOutputBlobsMemory(LocalCallOpKernelPhyInstrOperand* operand,
                                               DeviceCtx* device_ctx) {
    for (const auto& blob_object : *operand->outputs()) {
      blob_object->TryInitBlob();
      blob_object->TryAllocateBlobBodyMemory(device_ctx);
    }
  }

  static inline void TryAllocateTempStorage(LocalCallOpKernelPhyInstrOperand* operand,
                                            DeviceCtx* device_ctx) {
    if (operand->mut_call_ctx()->tmp_buffer_size > 0) {
      device_ctx->mut_allocator()->Allocate(&operand->mut_call_ctx()->tmp_buffer_ptr,
                                            operand->mut_call_ctx()->tmp_buffer_size);
    }
  }

  static inline void OpKernelCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                     DeviceCtx* device_ctx, user_op::OpKernelState* state,
                                     const user_op::OpKernelCache* cache) {
    auto* opkernel = operand->mut_opkernel();
    auto* compute_ctx = opkernel->GetComputeContext();
    OF_PROFILER_RANGE_PUSH("Compute");
    operand->user_opkernel()->Compute(compute_ctx, state, cache);
    OF_PROFILER_RANGE_POP();
  }

  static inline void DeallocateTempStorage(LocalCallOpKernelPhyInstrOperand* operand,
                                           DeviceCtx* device_ctx) {
    if (operand->mut_call_ctx()->tmp_buffer_size > 0) {
      CHECK_NOTNULL(operand->mut_call_ctx()->tmp_buffer_ptr);
      device_ctx->mut_allocator()->Deallocate(operand->mut_call_ctx()->tmp_buffer_ptr,
                                              operand->mut_call_ctx()->tmp_buffer_size);
    }
    operand->mut_call_ctx()->tmp_buffer_ptr = nullptr;
  }
};

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  auto* ptr = instruction->instr_msg().phy_instr_operand().get();
  auto* operand = CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(ptr));
  operand->WithThisCallContext([&] { LocalCallOpKernelUtil::Compute(instruction->instr_msg()); });
}

void LocalCallOpKernelInstructionType::ComputeInFuseMode(vm::InstructionMsg* instr_msg) const {
  auto* ptr = instr_msg->phy_instr_operand().get();
  auto* operand = CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(ptr));
  operand->WithThisCallContext([&] { LocalCallOpKernelUtil::Compute(*instr_msg); });
}

std::string LocalCallOpKernelInstructionType::DebugOpTypeName(
    const vm::InstructionMsg& instr_msg) const {
  auto* operand = CHECK_NOTNULL(instr_msg.phy_instr_operand().get());
  return CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand))
      ->opkernel()
      .op_type_name();
}

}  // namespace vm
}  // namespace oneflow
