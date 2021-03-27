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
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace vm {

class RankFrontSeqCallbackInstructionType : public InstructionType {
 public:
  RankFrontSeqCallbackInstructionType() = default;
  virtual ~RankFrontSeqCallbackInstructionType() override = default;

  bool IsFrontSequential() const override { return true; }

 protected:
  // clang-format off
  FLAT_MSG_VIEW_BEGIN(RankFrontSeqCallbackInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, process_rank);
  FLAT_MSG_VIEW_END(RankFrontSeqCallbackInstrOperand);
  // clang-format on

  void Run(const InstructionMsg& instr_msg) const {
    FlatMsgView<RankFrontSeqCallbackInstrOperand> args(instr_msg.operand());
    const auto& callback = instr_msg.no_arg_callback();
    if (args->process_rank() == GlobalProcessCtx::Rank()) {
      CHECK(static_cast<bool>(callback));
      (*callback)();
    } else {
      CHECK(!static_cast<bool>(callback));
    }
  }
};

class InferRankFrontSeqCallbackInstructionType final : public RankFrontSeqCallbackInstructionType {
 public:
  InferRankFrontSeqCallbackInstructionType() = default;
  ~InferRankFrontSeqCallbackInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override { Run(instruction->instr_msg()); }
  void Compute(Instruction* instruction) const override { /* do nothing */
  }
};
COMMAND(
    RegisterInstructionType<InferRankFrontSeqCallbackInstructionType>("InferRankFrontSeqCallback"));

class ComputeRankFrontSeqCallbackInstructionType final
    : public RankFrontSeqCallbackInstructionType {
 public:
  ComputeRankFrontSeqCallbackInstructionType() = default;
  ~ComputeRankFrontSeqCallbackInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { Run(instruction->instr_msg()); }
};
COMMAND(RegisterInstructionType<ComputeRankFrontSeqCallbackInstructionType>(
    "ComputeRankFrontSeqCallback"));

class CtrlInferRankFrontSeqCallbackInstructionType final
    : public RankFrontSeqCallbackInstructionType {
 public:
  CtrlInferRankFrontSeqCallbackInstructionType() = default;
  ~CtrlInferRankFrontSeqCallbackInstructionType() override = default;

  using stream_type = ControlStreamType;

  void Infer(VirtualMachine*, InstructionMsg* instr_msg) const override { Run(*instr_msg); }
  void Compute(VirtualMachine*, InstructionMsg* instr_msg) const override { /* do nothing */
    ;
  }
  void Infer(Instruction* instruction) const override { UNIMPLEMENTED(); }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<CtrlInferRankFrontSeqCallbackInstructionType>(
    "CtrlInferRankFrontSeqCallback"));

class CtrlComputeRankFrontSeqCallbackInstructionType final
    : public RankFrontSeqCallbackInstructionType {
 public:
  CtrlComputeRankFrontSeqCallbackInstructionType() = default;
  ~CtrlComputeRankFrontSeqCallbackInstructionType() override = default;

  using stream_type = ControlStreamType;

  void Infer(VirtualMachine*, InstructionMsg* instr_msg) const override { /* do nothing */
  }
  void Compute(VirtualMachine*, InstructionMsg* instr_msg) const override { Run(*instr_msg); }
  void Infer(Instruction* instruction) const override { UNIMPLEMENTED(); }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<CtrlComputeRankFrontSeqCallbackInstructionType>(
    "CtrlComputeRankFrontSeqCallback"));

class GlobalFrontSeqBarrierInstructionType : public InstructionType {
 public:
  GlobalFrontSeqBarrierInstructionType() = default;
  virtual ~GlobalFrontSeqBarrierInstructionType() override = default;

  using stream_type = HostStreamType;

  virtual bool IsFrontSequential() const override { return true; }

 protected:
  void Run() const { OF_ENV_BARRIER(); }
};

class InferGlobalFrontSeqBarrierInstructionType final
    : public GlobalFrontSeqBarrierInstructionType {
 public:
  InferGlobalFrontSeqBarrierInstructionType() = default;
  ~InferGlobalFrontSeqBarrierInstructionType() override = default;

  void Infer(Instruction* instruction) const override { Run(); }
  void Compute(Instruction* instruction) const override { /* do nothing */
  }
};
COMMAND(RegisterInstructionType<InferGlobalFrontSeqBarrierInstructionType>(
    "InferGlobalFrontSeqBarrier"));

class ComputeGlobalFrontSeqBarrierInstructionType final
    : public GlobalFrontSeqBarrierInstructionType {
 public:
  ComputeGlobalFrontSeqBarrierInstructionType() = default;
  ~ComputeGlobalFrontSeqBarrierInstructionType() override = default;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { Run(); }
};
COMMAND(RegisterInstructionType<ComputeGlobalFrontSeqBarrierInstructionType>(
    "ComputeGlobalFrontSeqBarrier"));

}  // namespace vm
}  // namespace oneflow
