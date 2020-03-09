#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/free_mirrored_object_handler.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"

namespace oneflow {

namespace {

class FreeMirroredObjectTryDeleter : public FreeMirroredObjectHandler {
 public:
  ~FreeMirroredObjectTryDeleter() override = default;

  void Call(LogicalObject* logical_object) const override {
    CHECK(!logical_object->is_zombie_link_empty());
    auto* scheduler = logical_object->mut_vm_scheduler();
    auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
    std::size_t size = parallel_id2mirrored_object->size();
    for (int i = 0; i < size; ++i) {
      auto* mirrored_object = parallel_id2mirrored_object->FindPtr(i);
      CHECK_NOTNULL(mirrored_object);
      if (!mirrored_object->access_list().empty()) { return; }
    }
    for (int i = 0; i < size; ++i) {
      auto* mirrored_object = parallel_id2mirrored_object->FindPtr(i);
      CHECK(!mirrored_object->has_object_type());
      parallel_id2mirrored_object->Erase(mirrored_object);
    }
    scheduler->mut_zombie_logical_object_list()->Erase(logical_object);
  }

  static const FreeMirroredObjectTryDeleter* Singleton() {
    static const FreeMirroredObjectTryDeleter singleton;
    return &singleton;
  }

 private:
  FreeMirroredObjectTryDeleter() : FreeMirroredObjectHandler() {}
};

}  // namespace

enum CtrlInstrOpCode { kNewMirroredObjectSymbol = 0, kDeleteMirroredObjectSymbol };

typedef void (*CtrlInstrFunc)(VmScheduler*, VmInstructionMsg*);
std::vector<CtrlInstrFunc> ctrl_instr_table;

#define REGISTER_CTRL_INSTRUCTION(op_code, function_name) \
  COMMAND({                                               \
    ctrl_instr_table.resize(op_code + 1);                 \
    ctrl_instr_table.at(op_code) = &function_name;        \
  })

// clang-format off
FLAT_MSG_VIEW_BEGIN(NewMirroredObjectSymbolCtrlInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(LogicalObjectId, logical_object_id);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, parallel_num);
FLAT_MSG_VIEW_END(NewMirroredObjectSymbolCtrlInstruction);
// clang-format on

ObjectMsgPtr<VmInstructionMsg> ControlVmStreamType::NewMirroredObjectSymbol(
    const LogicalObjectId& logical_object_id, int64_t parallel_num) const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(CtrlInstrOpCode::kNewMirroredObjectSymbol);
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  {
    FlatMsgView<NewMirroredObjectSymbolCtrlInstruction> view(vm_instr_proto->mutable_operand());
    view->set_logical_object_id(logical_object_id);
    view->set_parallel_num(parallel_num);
  }
  return vm_instr_msg;
}

void NewMirroredObjectSymbol(VmScheduler* scheduler, VmInstructionMsg* vm_instr_msg) {
  FlatMsgView<NewMirroredObjectSymbolCtrlInstruction> view;
  CHECK(view->Match(vm_instr_msg->mut_vm_instruction_proto()->mut_operand()));
  auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(
      scheduler->mut_scheduler_thread_only_allocator(), view->logical_object_id(), scheduler);
  CHECK(scheduler->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
  auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
  for (int64_t i = 0; i < view->parallel_num(); ++i) {
    auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(scheduler->mut_allocator(),
                                                                 logical_object.Mutable(), i);
    CHECK(parallel_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
  }
}
REGISTER_CTRL_INSTRUCTION(CtrlInstrOpCode::kNewMirroredObjectSymbol, NewMirroredObjectSymbol);

// clang-format off
FLAT_MSG_VIEW_BEGIN(DeleteMirroredObjectSymbolCtrlInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
FLAT_MSG_VIEW_END(DeleteMirroredObjectSymbolCtrlInstruction);
// clang-format on

const VmStreamTypeId ControlVmStreamType::kVmStreamTypeId;

ObjectMsgPtr<VmInstructionMsg> ControlVmStreamType::DeleteMirroredObjectSymbol(
    const LogicalObjectId& logical_object_id) const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(CtrlInstrOpCode::kDeleteMirroredObjectSymbol);
  {
    FlatMsgView<DeleteMirroredObjectSymbolCtrlInstruction> view(vm_instr_proto->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
  }
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  return vm_instr_msg;
}
void DeleteMirroredObjectSymbol(VmScheduler* scheduler, VmInstructionMsg* vm_instr_msg) {
  FlatMsgView<DeleteMirroredObjectSymbolCtrlInstruction> view;
  CHECK(view->Match(vm_instr_msg->mut_vm_instruction_proto()->mut_operand()));
  const auto& logical_objectId = view->mirrored_object_operand().operand().logical_object_id();
  auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_objectId);
  CHECK_NOTNULL(logical_object);
  CHECK(logical_object->is_zombie_link_empty());
  scheduler->mut_zombie_logical_object_list()->PushBack(logical_object);
  scheduler->mut_id2logical_object()->Erase(logical_object);
  logical_object->set_free_mirrored_object_handler(FreeMirroredObjectTryDeleter::Singleton());
  logical_object->free_mirrored_object_handler().Call(logical_object);
}
REGISTER_CTRL_INSTRUCTION(CtrlInstrOpCode::kDeleteMirroredObjectSymbol, DeleteMirroredObjectSymbol);

void ControlVmStreamType::Run(VmScheduler* scheduler, VmInstructionMsg* vm_instr_msg) const {
  VmInstructionOpcode opcode = vm_instr_msg->vm_instruction_proto().opcode();
  return ctrl_instr_table.at(opcode)(scheduler, vm_instr_msg);
}

}  // namespace oneflow
