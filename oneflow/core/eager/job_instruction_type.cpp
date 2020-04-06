#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

COMMAND(Global<vm::Storage<Job>>::SetAllocated(new vm::Storage<Job>()));

namespace {

class NewJobObjectInstructionType final : public vm::InstructionType {
 public:
  NewJobObjectInstructionType() = default;
  ~NewJobObjectInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewJobObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutableOperand, job);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, job_id);
  FLAT_MSG_VIEW_END(NewJobObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<NewJobObjectInstrOperand> view(instr_ctx->instr_msg().operand());
    const auto& job = Global<vm::Storage<Job>>::Get()->Get(view->job().logical_object_id());
    instr_ctx->mut_operand_type(view->job())->Mutable<JobObject>(job, view->job_id());
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<NewJobObjectInstructionType>("NewJobObject"));
COMMAND(vm::RegisterLocalInstructionType<NewJobObjectInstructionType>("NewLocalJobObject"));

class DeleteJobObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteJobObjectInstructionType() = default;
  ~DeleteJobObjectInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteJobObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutableOperand, job);
  FLAT_MSG_VIEW_END(DeleteJobObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<DeleteJobObjectInstrOperand> view(instr_ctx->instr_msg().operand());
    auto* type_mirrored_object = instr_ctx->mut_operand_type(view->job());
    CHECK(type_mirrored_object->Has<JobObject>());
    type_mirrored_object->reset_object();
    Global<vm::Storage<Job>>::Get()->Clear(view->job().logical_object_id());
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteJobObjectInstructionType>("DeleteJobObject"));
COMMAND(vm::RegisterLocalInstructionType<DeleteJobObjectInstructionType>("DeleteLocalJobObject"));

}  // namespace

}  // namespace eager
}  // namespace oneflow
