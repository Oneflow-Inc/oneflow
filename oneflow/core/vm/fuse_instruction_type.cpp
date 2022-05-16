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
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/fuse_phy_instr_operand.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace vm {

template<typename StreamT>
class FuseInstructionType : public vm::InstructionType {
 public:
  FuseInstructionType() = default;
  ~FuseInstructionType() override = default;

  using stream_type = StreamT;

  std::string DebugOpTypeName(const InstructionMsg&) const override { return "Fuse"; }

  void InitInstructionStatus(Instruction* instruction) const override {
    const auto& phy_instr_operand = instruction->instr_msg().phy_instr_operand();
    auto* ptr = dynamic_cast<vm::FusePhyInstrOperand*>(phy_instr_operand.get());
    auto* instr_msg_list = CHECK_NOTNULL(ptr)->mut_instr_msg_list();
    auto* last_instr_msg = CHECK_NOTNULL(instr_msg_list->Last());
    // init instruction status by last instruction_msg.
    last_instr_msg->instr_type_id().instruction_type().InitInstructionStatusIf(instruction);
  }

  void Compute(vm::Instruction* instruction) const override {
    const auto& phy_instr_operand = instruction->instr_msg().phy_instr_operand();
    auto* ptr = dynamic_cast<vm::FusePhyInstrOperand*>(phy_instr_operand.get());
    auto* instr_msg_list = CHECK_NOTNULL(ptr)->mut_instr_msg_list();
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instr_msg, instr_msg_list) {
      OF_PROFILER_RANGE_GUARD("F:" + instr_msg->DebugName());
      instr_msg->instr_type_id().instruction_type().ComputeInFuseMode(instr_msg);
    }
  }
};

COMMAND(vm::RegisterInstructionType<FuseInstructionType<CpuStreamType>>("cpu.Fuse"));
COMMAND(vm::RegisterInstructionType<FuseInstructionType<CpuStreamType>>("comm_net.Fuse"));

#ifdef WITH_CUDA
COMMAND(vm::RegisterInstructionType<FuseInstructionType<CudaStreamType>>("cuda.Fuse"));
COMMAND(vm::RegisterInstructionType<FuseInstructionType<CudaCopyH2DStreamType>>("cuda_h2d.Fuse"));
COMMAND(vm::RegisterInstructionType<FuseInstructionType<CudaCopyD2HStreamType>>("cuda_d2h.Fuse"));
COMMAND(
    vm::RegisterInstructionType<FuseInstructionType<CudaStreamType>>("sync_launched_nccl.Fuse"));
COMMAND(vm::RegisterInstructionType<FuseInstructionType<AsyncCudaStreamType>>(
    "async_launched_nccl.Fuse"));
#endif

}  // namespace vm
}  // namespace oneflow
