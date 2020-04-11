#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

class NewObjectInstructionType final : public InstructionType {
 public:
  NewObjectInstructionType() = default;
  ~NewObjectInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewObjectInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(ConstHostOperand, parallel_desc);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, logical_object_id);
  FLAT_MSG_VIEW_END(NewObjectInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run<&GetTypeLogicalObjectId>(scheduler, instr_msg);
  }
  void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run<&GetSelfLogicalObjectId>(scheduler, instr_msg);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(Scheduler* scheduler, InstructionMsg* instr_msg) const {
    TODO();
  }
};
COMMAND(RegisterInstructionType<NewObjectInstructionType>("NewObject"));
COMMAND(RegisterLocalInstructionType<NewObjectInstructionType>("LocalNewObject"));

class DeleteSymbolInstructionType final : public InstructionType {
 public:
  DeleteSymbolInstructionType() = default;
  ~DeleteSymbolInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteSymbolCtrlInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(MutOperand, symbol);
  FLAT_MSG_VIEW_END(DeleteSymbolCtrlInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    // do nothing, delete symbol in Compute method
    Run<&GetTypeLogicalObjectId>(scheduler, instr_msg);
  }
  void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run<&GetSelfLogicalObjectId>(scheduler, instr_msg);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(Scheduler* scheduler, InstructionMsg* instr_msg) const {
    FlatMsgView<DeleteSymbolCtrlInstruction> view;
    CHECK(view.Match(instr_msg->operand()));
    FOR_RANGE(int, i, 0, view->symbol_size()) {
      int64_t logical_object_id = view->symbol(i).operand().logical_object_id();
      logical_object_id = GetLogicalObjectId(logical_object_id);
      auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_object_id);
      CHECK_NOTNULL(logical_object);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      for (int global_device_id = 0; global_device_id < global_device_id2mirrored_object->size();
           ++global_device_id) {
        auto* mirrored_object = global_device_id2mirrored_object->FindPtr(global_device_id);
        CHECK(!mirrored_object->has_object());
        global_device_id2mirrored_object->Erase(mirrored_object);
      }
      scheduler->mut_id2logical_object()->Erase(logical_object);
    }
  }
};
COMMAND(RegisterInstructionType<DeleteSymbolInstructionType>("DeleteObject"));
COMMAND(RegisterLocalInstructionType<DeleteSymbolInstructionType>("LocalDeleteObject"));

}  // namespace vm
}  // namespace oneflow
