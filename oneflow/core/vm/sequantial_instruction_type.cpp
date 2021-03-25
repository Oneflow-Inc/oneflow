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
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace vm {

class RankSequantialCallbackInstructionType : public InstructionType {
 public:
  RankSequantialCallbackInstructionType() = default;
  virtual ~RankSequantialCallbackInstructionType() override = default;

  using stream_type = ControlStreamType;

  virtual bool IsSequantialInstructionType() const { return true; }

  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 protected:
  // clang-format off
  FLAT_MSG_VIEW_BEGIN(RankSequantialCallbackInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, process_rank);
  FLAT_MSG_VIEW_END(RankSequantialCallbackInstrOperand);
  // clang-format on

  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<RankSequantialCallbackInstrOperand> args(instr_msg->operand());
    const auto& callback = instr_msg->sequantial_callback();
    if (args->process_rank() == GlobalProcessCtx::Rank()) {
      CHECK(static_cast<bool>(callback));
      (*callback)();
    } else {
      CHECK(!static_cast<bool>(callback));
    }
  }
};

class RankSequantialInferCallbackInstructionType final
    : public RankSequantialCallbackInstructionType {
 public:
  RankSequantialInferCallbackInstructionType() = default;
  ~RankSequantialInferCallbackInstructionType() override = default;

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override { Run(vm, instr_msg); }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override { /* do nothing */
  }
};
COMMAND(RegisterInstructionType<RankSequantialInferCallbackInstructionType>(
    "RankSequantialInferCallback"));

class RankSequantialComputeCallbackInstructionType final
    : public RankSequantialCallbackInstructionType {
 public:
  RankSequantialComputeCallbackInstructionType() = default;
  ~RankSequantialComputeCallbackInstructionType() override = default;

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override { /* do nothing */
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override { Run(vm, instr_msg); }
};
COMMAND(RegisterInstructionType<RankSequantialComputeCallbackInstructionType>(
    "RankSequantialComputeCallback"));

}  // namespace vm
}  // namespace oneflow
