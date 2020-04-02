#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/eager/op_object.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class NewOpObjectInstructionType final : public vm::InstructionType {
 public:
  NewOpObjectInstructionType() = default;
  ~NewOpObjectInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewOpObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::ConstMirroredObjectOperand, job);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableMirroredObjectOperand, op);
  FLAT_MSG_VIEW_END(NewOpObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<NewOpObjectInstrOperand> view;
    CHECK(view->Match(instr_ctx->mut_instr_msg()->mut_operand()));
    const auto& job_object = instr_ctx->operand_type(view->job()).Get<JobObject>();
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const OperatorConf& op_conf = job_object.LookupOpConf(view->op(i).logical_object_id());
      instr_ctx->mut_operand_type(view->op(i))->Mutable<OpObject>(op_conf, &job_object.job_desc());
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<NewOpObjectInstructionType>("NewOpObject"));
COMMAND(vm::RegisterLocalInstructionType<NewOpObjectInstructionType>("NewLocalOpObject"));

class DeleteOpObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteOpObjectInstructionType() = default;
  ~DeleteOpObjectInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteOpObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableMirroredObjectOperand, op);
  FLAT_MSG_VIEW_END(DeleteOpObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<DeleteOpObjectInstrOperand> view;
    CHECK(view->Match(instr_ctx->mut_instr_msg()->mut_operand()));
    for (int i = 0; i < view->op_size(); ++i) {
      auto* type_mirrored_object = instr_ctx->mut_operand_type(view->op(i));
      CHECK(type_mirrored_object->Has<OpObject>());
      type_mirrored_object->reset_object();
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteOpObjectInstructionType>("DeleteOpObject"));
COMMAND(vm::RegisterLocalInstructionType<DeleteOpObjectInstructionType>("DeleteLocalOpObject"));

}  // namespace eager
}  // namespace oneflow
