#include "oneflow/core/vm/vpu_type_desc.msg.h"
#include "oneflow/core/vm/control_vpu.h"
#include "oneflow/core/vm/vpu_instruction.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/static_counter.h"
#include "oneflow/core/common/flat_msg_view.h"

namespace oneflow {

static const VpuTypeId kControlVpuTypeId = 0;
typedef void (*CtrlInstrFunc)(VpuSchedulerCtx*, VpuInstructionMsg*);
std::vector<CtrlInstrFunc> ctrl_instr_table;
template<int opcode>
struct CtrlInstruction;
DEFINE_STATIC_COUNTER(kCtrlInstrOpCode);
#define REGISTER_CTRL_INSTRUCTION(op_code, function_name) \
  COMMAND({                                               \
    ctrl_instr_table.resize(op_code + 1);                 \
    ctrl_instr_table.at(op_code) = &function_name;        \
  })

static const int kNewMirroredObjectSymbol = STATIC_COUNTER(kCtrlInstrOpCode);
// clang-format off
template<>
BEGIN_FLAT_MSG_VIEW(CtrlInstruction<kNewMirroredObjectSymbol>);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, symbol);
  FLAT_MSG_VIEW_DEFINE_PATTERN(bool, is_remote);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, parallel_num);
END_FLAT_MSG_VIEW(CtrlInstruction<kNewMirroredObjectSymbol>);
// clang-format on

void MakeLogicalObjectId(LogicalObjectId* logical_object_id, uint64_t symbol, bool is_remote) {
  if (is_remote) {
    logical_object_id->set_remote_value(symbol);
  } else {
    logical_object_id->set_local_value(symbol);
  }
}

void NewMirroredObjectSymbol(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) {
  FlatMsgView<CtrlInstruction<kNewMirroredObjectSymbol>> view;
  CHECK(view->Match(vpu_instr_msg->mut_vpu_instruction_proto()->mut_operand()));
  FlatMsg<LogicalObjectId> logical_object_id;
  MakeLogicalObjectId(logical_object_id.Mutable(), view->symbol(), view->is_remote());
  auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(scheduler->mut_default_allocator(),
                                                             logical_object_id.Get());
  CHECK(scheduler->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
  auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
  for (int64_t i = 0; i < view->parallel_num(); ++i) {
    auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(scheduler->mut_default_allocator(),
                                                                 logical_object.Get(), i);
    CHECK(parallel_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
  }
}
REGISTER_CTRL_INSTRUCTION(kNewMirroredObjectSymbol, NewMirroredObjectSymbol);
ObjectMsgPtr<VpuInstructionMsg> ControlVpu::NewMirroredObjectSymbol(uint64_t symbol, bool is_remote,
                                                                    int64_t parallel_num) const {
  auto vpu_instr_msg = ObjectMsgPtr<VpuInstructionMsg>::New();
  auto* vpu_instr_proto = vpu_instr_msg->mut_vpu_instruction_proto();
  vpu_instr_proto->set_vpu_type_id(kControlVpuTypeId);
  vpu_instr_proto->set_opcode(kNewMirroredObjectSymbol);
  {
    FlatMsgView<CtrlInstruction<kNewMirroredObjectSymbol>> view(vpu_instr_proto->mut_operand());
    view->set_symbol(symbol);
    view->set_is_remote(is_remote);
    view->set_parallel_num(parallel_num);
  }
  vpu_instr_proto->mut_vpu_mask()->mut_all_vpu_enabled();
  return vpu_instr_msg;
}

INCREASE_STATIC_COUNTER(kCtrlInstrOpCode);

static const int kDeleteMirroredObjectSymbol = STATIC_COUNTER(kCtrlInstrOpCode);
// clang-format off
template<>
BEGIN_FLAT_MSG_VIEW(CtrlInstruction<kDeleteMirroredObjectSymbol>);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableLogicalObjectId, mutable_logical_object_id);
END_FLAT_MSG_VIEW(CtrlInstruction<kDeleteMirroredObjectSymbol>);
// clang-format on

void DeleteMirroredObjectSymbol(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) {
  FlatMsgView<CtrlInstruction<kDeleteMirroredObjectSymbol>> view;
  CHECK(view->Match(vpu_instr_msg->mut_vpu_instruction_proto()->mut_operand()));
  const auto& logical_objectId = view->mutable_logical_object_id().value();
  auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_objectId);
  CHECK_NOTNULL(logical_object);
  auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
  std::size_t size = parallel_id2mirrored_object->size();
  for (int i = 0; i < size; ++i) {
    auto* mirrored_object = parallel_id2mirrored_object->FindPtr(i);
    CHECK_NOTNULL(mirrored_object);
    CHECK(mirrored_object->is_maybe_available_access_link_empty());
    CHECK(mirrored_object->waiting_access_list().empty());
    CHECK(mirrored_object->holding_access_list().empty());
    parallel_id2mirrored_object->Erase(mirrored_object);
  }
  scheduler->mut_id2logical_object()->Erase(logical_object);
}
REGISTER_CTRL_INSTRUCTION(kDeleteMirroredObjectSymbol, DeleteMirroredObjectSymbol);

ObjectMsgPtr<VpuInstructionMsg> ControlVpu::DeleteMirroredObjectSymbol(
    const LogicalObjectId& logical_object_id) const {
  auto vpu_instr_msg = ObjectMsgPtr<VpuInstructionMsg>::New();
  auto* vpu_instr_proto = vpu_instr_msg->mut_vpu_instruction_proto();
  vpu_instr_proto->set_vpu_type_id(kControlVpuTypeId);
  vpu_instr_proto->set_opcode(kDeleteMirroredObjectSymbol);
  {
    FlatMsgView<CtrlInstruction<kDeleteMirroredObjectSymbol>> view(vpu_instr_proto->mut_operand());
    view->mut_mutable_logical_object_id()->mut_value()->CopyFrom(logical_object_id);
  }
  vpu_instr_proto->mut_vpu_mask()->mut_all_vpu_enabled();
  return vpu_instr_msg;
}

void ControlVpu::Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const {
  VpuInstructionOpcode opcode = vpu_instr_msg->vpu_instruction_proto().opcode();
  return ctrl_instr_table.at(opcode)(scheduler, vpu_instr_msg);
}

}  // namespace oneflow
