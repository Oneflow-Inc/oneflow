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
#include "oneflow/core/eager/release_tensor_instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"

namespace oneflow {

namespace vm {

void ReleaseTensorInstructionType::Infer(vm::Instruction* instruction) const { UNIMPLEMENTED(); }

void ReleaseTensorInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  CHECK_JUST(ptr->eager_blob_object()->DeallocateBlobDataPtr());
}

class CpuReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  CpuReleaseTensorInstructionType() = default;
  ~CpuReleaseTensorInstructionType() override = default;
  using stream_type = vm::CpuStreamType;
};
COMMAND(vm::RegisterInstructionType<CpuReleaseTensorInstructionType>("cpu.ReleaseTensor"));

#ifdef WITH_CUDA
class GpuReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  GpuReleaseTensorInstructionType() = default;
  ~GpuReleaseTensorInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuReleaseTensorInstructionType>("gpu.ReleaseTensor"));
#endif

}  // namespace vm
}  // namespace oneflow
