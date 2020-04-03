#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class NewOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  NewOpKernelObjectInstructionType() = default;
  ~NewOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewOpKernelObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::ConstOperand, job);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableOperand, op);
  FLAT_MSG_VIEW_END(NewOpKernelObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<NewOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    const auto& job_object = instr_ctx->operand_type(view->job()).Get<JobObject>();
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const OperatorConf& op_conf =
          job_object.OpConf4LogicalObjectId(view->op(i).logical_object_id());
      instr_ctx->mut_operand_type(view->op(i))
          ->Mutable<OpKernelObject>(job_object.job_desc(), op_conf);
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<NewOpKernelObjectInstructionType>("NewOpKernelObject"));
COMMAND(
    vm::RegisterLocalInstructionType<NewOpKernelObjectInstructionType>("NewLocalOpKernelObject"));

class DeleteOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteOpKernelObjectInstructionType() = default;
  ~DeleteOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteOpKernelObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableOperand, op);
  FLAT_MSG_VIEW_END(DeleteOpKernelObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<DeleteOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    for (int i = 0; i < view->op_size(); ++i) {
      auto* type_mirrored_object = instr_ctx->mut_operand_type(view->op(i));
      CHECK(type_mirrored_object->Has<OpKernelObject>());
      type_mirrored_object->reset_object();
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteOpKernelObjectInstructionType>("DeleteOpKernelObject"));
COMMAND(vm::RegisterLocalInstructionType<DeleteOpKernelObjectInstructionType>(
    "DeleteLocalOpKernelObject"));

}  // namespace eager
}  // namespace oneflow
