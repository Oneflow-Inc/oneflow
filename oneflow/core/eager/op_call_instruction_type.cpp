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
#include "oneflow/core/profiler/collection.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace vm {

struct OpCallInstructionUtil final {
  static inline Maybe<void> Compute(const vm::Instruction& instruction) {
    auto* operand = GetCallPhyInstrOperand(instr_msg);
    DeviceCtx* device_ctx = instr_msg.stream().device_ctx().get();
    OF_PROFILER_RANGE_PUSH("AllocateOutputBlobsMemory");
    JUST(AllocateOutputBlobsMemory(operand, device_ctx));
    OF_PROFILER_RANGE_POP();
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_GUARD("TryAllocateTempStorageBlobMemory");
      InferTempStorageBlobDesc(operand);
      JUST(TryAllocateTempStorageBlobMemory(operand, device_ctx));
    }
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (operand->user_opkernel()->has_state_or_cache()) {
      OF_PROFILER_RANGE_GUARD("TryInitOpKernelStateAndCache");
      TryInitOpKernelStateAndCache(operand, device_ctx, &state, &cache);
    }
    OpKernelCompute(operand, device_ctx, state, cache);
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_GUARD("DeallocateTempStorageBlobMemory");
      JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
    }
    return Maybe<void>::Ok();
  }

  static inline OpCallPhyInstrOperand* GetCallPhyInstrOperand(const vm::Instruction& instruction) {
    auto* operand = CHECK_NOTNULL(instruction.phy_instr_operand().get());
    return CHECK_NOTNULL(dynamic_cast<OpCallPhyInstrOperand*>(operand));
  }

 private:
  static inline void InferTempStorageBlobDesc(OpCallPhyInstrOperand* operand) {
    auto* temp_eager_blob_object = operand->mut_opkernel()->mut_temp_blob_object();
    CHECK(temp_eager_blob_object->data_type() == DataType::kChar);
    size_t temp_size =
        operand->opkernel().InferTmpSize(&operand->call_ctx_, operand->user_opkernel());
    *temp_eager_blob_object->mut_shape() = Shape({static_cast<int64_t>(temp_size)});
    *temp_eager_blob_object->mut_stride() = Stride(*temp_eager_blob_object->mut_shape());
    temp_eager_blob_object->set_pin_memory(false);
    temp_eager_blob_object->set_is_dynamic(true);
  }

  static inline void TryInitOpKernelStateAndCache(OpCallPhyInstrOperand* operand,
                                                  DeviceCtx* device_ctx,
                                                  user_op::OpKernelState** state,
                                                  user_op::OpKernelCache** cache) {
    if (likely(operand->op_interp_ctx().state)) {
      *state = operand->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    operand->mut_opkernel()->TryInitOpKernelStateAndCache(&operand->call_ctx_, device_ctx,
                                                          operand->user_opkernel(), state, cache);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(OpCallPhyInstrOperand* operand,
                                                      DeviceCtx* device_ctx) {
    for (const auto& blob_object : *operand->outputs()) {
      JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorageBlobMemory(OpCallPhyInstrOperand* operand,
                                                             DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->TryAllocateBlobBodyMemory(device_ctx);
  }

  static inline void OpKernelCompute(OpCallPhyInstrOperand* operand, DeviceCtx* device_ctx,
                                     user_op::OpKernelState* state,
                                     const user_op::OpKernelCache* cache) {
    auto* call_ctx = &operand->call_ctx_;
    auto* user_kernel = operand->user_opkernel();
    operand->mut_opkernel()->Compute(call_ctx, device_ctx, user_kernel, state, cache);
  }

  static inline Maybe<void> DeallocateTempStorageBlobMemory(OpCallPhyInstrOperand* operand,
                                                            DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->DeallocateBlobDataPtr();
  }
};

void OpCallInstructionType::Compute(vm::Instruction* instruction) const {
  CHECK_JUST(OpCallInstructionUtil::Compute(*instruction));
}

std::string OpCallInstructionType::DebugName(const vm::Instruction& instruction) const {
  auto* operand = CHECK_NOTNULL(instruction.phy_instr_operand().get());
  return CHECK_NOTNULL(dynamic_cast<OpCallPhyInstrOperand*>(operand))->opkernel().op_type_name()
         + ":OpCall";
}

}  // namespace vm
}  // namespace oneflow
