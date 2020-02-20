#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

using WaitingVmInstrCtxList = VmScheduler::waiting_vm_instr_ctx_list_ObjectMsgListType;
using ReadyVmInstrCtxList = OBJECT_MSG_LIST(VmInstructionCtx, vm_instruction_ctx_link);
using MaybeAvailableAccessList = OBJECT_MSG_LIST(MirroredObject, maybe_available_access_link);
using TmpWaitingVmInstrMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instruction_msg_link);
using NewVmInstrCtxList = VmScheduler::new_vm_instr_ctx_list_ObjectMsgListType;
using Id2LogicalObject = VmScheduler::id2logical_object_ObjectMsgSkipListType;
using ActiveVmStreamList = VmScheduler::active_vm_stream_list_ObjectMsgListType;

namespace {

void ReleaseVmInstructionCtx(VmInstructionCtx* vm_instr_ctx,
                             /*out*/ MaybeAvailableAccessList* maybe_available_access_list) {
  auto* holding_operand_list = vm_instr_ctx->mut_holding_operand_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(holding_operand_list, mirrored_object_access) {
    auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
    auto* holding_access_list = mirrored_object->mut_holding_access_list();
    holding_access_list->Erase(mirrored_object_access);
    if (!holding_access_list->empty()) { continue; }
    mirrored_object->clear_current_access_type();
    if (mirrored_object->waiting_access_list().empty()) {
      mirrored_object->logical_object().free_mirrored_object_handler().Call(
          mirrored_object->mut_logical_object());
      continue;
    }
    maybe_available_access_list->PushBack(mirrored_object);
  }
}

void ReleaseVmInstructionPackage(VmInstructionPackage* pkg,
                                 /*out*/ MaybeAvailableAccessList* maybe_available_access_list) {
  auto* vm_instr_ctx_list = pkg->mut_vm_instruction_ctx_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_ctx_list, vm_instr_ctx) {
    ReleaseVmInstructionCtx(vm_instr_ctx, /*out*/ maybe_available_access_list);
    vm_instr_ctx_list->Erase(vm_instr_ctx);
  }
}

void TryReleaseFinishedVmInstructionPackages(
    VmStream* vm_stream, /*out*/ MaybeAvailableAccessList* maybe_available_access_list) {
  auto* pkg_list = vm_stream->mut_running_pkg_list();
  while (true) {
    auto* begin = pkg_list->Begin();
    if (begin == nullptr || !begin->Done()) { break; }
    ReleaseVmInstructionPackage(begin, /*out*/ maybe_available_access_list);
    pkg_list->Erase(begin);
  }
}

void FilterReadyVmInstrCtx(MaybeAvailableAccessList* maybe_available_access_list,
                           WaitingVmInstrCtxList* waiting_vm_instr_ctx_list,
                           /*out*/ ReadyVmInstrCtxList* ready_vm_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(maybe_available_access_list, mirrored_object) {
    mirrored_object->TryResetCurrentAccessType();
    auto* waiting_access_list = mirrored_object->mut_waiting_access_list();
    auto* holding_access_list = mirrored_object->mut_holding_access_list();
    while (auto* mirrored_object_access = mirrored_object->GetFirstAllowedAccess()) {
      waiting_access_list->MoveToDstBack(mirrored_object_access, holding_access_list);
      auto* vm_instruction_ctx = mirrored_object_access->mut_vm_instruction_ctx();
      auto* waiting_operand_list = vm_instruction_ctx->mut_waiting_operand_list();
      auto* holding_operand_list = vm_instruction_ctx->mut_holding_operand_list();
      waiting_operand_list->MoveToDstBack(mirrored_object_access, holding_operand_list);
      if (waiting_operand_list->empty()) {
        waiting_vm_instr_ctx_list->MoveToDstBack(vm_instruction_ctx, ready_vm_instr_ctx_list);
      }
    }
    maybe_available_access_list->Erase(mirrored_object);
  }
}

void FilterAndRunControlVmInstructions(VmScheduler* scheduler,
                                       TmpWaitingVmInstrMsgList* vm_instr_msg_list) {
  ControlVmStreamType control_vm_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    const VmStreamTypeId vm_stream_type_id =
        vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    if (vm_stream_type_id != kControlVmStreamTypeId) { continue; }
    control_vm_stream_type.Run(scheduler, vm_instr_msg);
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

void MakeVmInstructionCtx(VmScheduler* scheduler, TmpWaitingVmInstrMsgList* vm_instr_msg_list,
                          /*out*/ NewVmInstrCtxList* ret_vm_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    VmStreamTypeId vm_stream_type_id = vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    auto* vm_stream_rt_desc =
        scheduler->mut_vm_stream_type_id2vm_stream_rt_desc()->FindPtr(vm_stream_type_id);
    OBJECT_MSG_LIST_FOR_EACH_UNSAFE_PTR(vm_stream_rt_desc->mut_vm_stream_list(), vm_stream) {
      auto vm_instr_ctx = ObjectMsgPtr<VmInstructionCtx>::NewFrom(
          scheduler->mut_default_allocator(), vm_instr_msg, vm_stream);
      ret_vm_instr_ctx_list->PushBack(vm_instr_ctx.Mutable());
    }
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

MirroredObject* FindMirroredObject(Id2LogicalObject* id2logical_object,
                                   const LogicalObjectId& logical_object_id, int64_t parallel_id) {
  auto* logical_object = id2logical_object->FindPtr(logical_object_id);
  CHECK_NOTNULL(logical_object);
  auto* ret = logical_object->mut_parallel_id2mirrored_object()->FindPtr(parallel_id);
  CHECK_NOTNULL(ret);
  return ret;
}

static const bool kConstOperandAccess = true;
static const bool kMutableOperandAccess = false;
template<bool is_const_operand>
void ConsumeMirroredObject(MirroredObject* mirrored_object, VmInstructionCtx* vm_instr_ctx) {
  uint64_t id_value = mirrored_object->logical_object().logical_object_id().value();
  auto mirrored_object_access = ObjectMsgPtr<MirroredObjectAccess>::NewFrom(
      vm_instr_ctx->mut_allocator(), vm_instr_ctx, mirrored_object, id_value, is_const_operand);
  mirrored_object->mut_waiting_access_list()->PushBack(mirrored_object_access.Mutable());
  vm_instr_ctx->mut_waiting_operand_list()->PushBack(mirrored_object_access.Mutable());
  vm_instr_ctx->mut_logical_object_id2operand_access()->Insert(mirrored_object_access.Mutable());
}

inline void TryPushBack(MaybeAvailableAccessList* maybe_available_access_list,
                        MirroredObject* mirrored_object) {
  if (mirrored_object->is_maybe_available_access_link_empty()) {
    maybe_available_access_list->PushBack(mirrored_object);
  }
}

void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                            NewVmInstrCtxList* new_vm_instr_ctx_list,
                            /*out*/ MaybeAvailableAccessList* maybe_available_access_list) {
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE_PTR(new_vm_instr_ctx_list, vm_instr_ctx) {
    int64_t parallel_id = vm_instr_ctx->vm_stream().vm_stream_id().parallel_id();
    const auto& operands = vm_instr_ctx->vm_instruction_msg().vm_instruction_proto().operand();
    for (const auto& operand : operands) {
      if (operand.has_const_operand()) {
        auto* mirrored_object =
            FindMirroredObject(id2logical_object, operand.const_operand().value(), parallel_id);
        ConsumeMirroredObject<kConstOperandAccess>(mirrored_object, vm_instr_ctx);
        TryPushBack(maybe_available_access_list, mirrored_object);
      } else if (operand.has_mutable_operand()) {
        auto* mirrored_object =
            FindMirroredObject(id2logical_object, operand.mutable_operand().value(), parallel_id);
        ConsumeMirroredObject<kMutableOperandAccess>(mirrored_object, vm_instr_ctx);
        TryPushBack(maybe_available_access_list, mirrored_object);
      } else {
        // do nothing
      }
    }
  }
}

void MoveToReadyCtxListIfNoObjectOperand(NewVmInstrCtxList* new_vm_instr_ctx_list,
                                         /*out*/ ReadyVmInstrCtxList* ready_vm_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_vm_instr_ctx_list, vm_instr_ctx) {
    if (vm_instr_ctx->waiting_operand_list().empty()) {
      new_vm_instr_ctx_list->MoveToDstBack(vm_instr_ctx, ready_vm_instr_ctx_list);
    }
  }
}

void DispatchVmInstructionCtx(VmScheduler* scheduler,
                              ReadyVmInstrCtxList* ready_vm_instr_ctx_list) {
  auto* allocator = scheduler->mut_default_allocator();
  OBJECT_MSG_LIST(VmStream, tmp_active_vm_stream_link) tmp_active_vm_stream_list;
  while (auto* first = ready_vm_instr_ctx_list->Begin()) {
    auto* vm_stream = first->mut_vm_stream();
    ready_vm_instr_ctx_list->MoveToDstBack(first, vm_stream->mut_collect_vm_instruction_list());
    if (vm_stream->is_tmp_active_vm_stream_link_empty()) {
      tmp_active_vm_stream_list.PushBack(vm_stream);
    }
  }
  auto* active_vm_stream_list = scheduler->mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_active_vm_stream_list, vm_stream) {
    tmp_active_vm_stream_list.Erase(vm_stream);
    auto pkg = ObjectMsgPtr<VmInstructionPackage>::NewFrom(allocator, vm_stream);
    vm_stream->mut_collect_vm_instruction_list()->MoveTo(pkg->mut_vm_instruction_ctx_list());
    vm_stream->mut_running_pkg_list()->PushBack(pkg.Mutable());
    if (vm_stream->is_active_vm_stream_link_empty()) { active_vm_stream_list->PushBack(vm_stream); }
    vm_stream->mut_waiting_pkg_list()->EmplaceBack(std::move(pkg));
  }
}

}  // namespace

void VmScheduler::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  set_default_allocator(allocator);
  TODO();
}

void VmScheduler::Receive(VmInstructionMsgList* vm_instr_list) {
  mut_waiting_msg_list()->MoveFrom(vm_instr_list);
}

void VmScheduler::Schedule() {
  MaybeAvailableAccessList maybe_available_access_list;
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_vm_stream_list, vm_stream) {
    TryReleaseFinishedVmInstructionPackages(vm_stream, /*out*/ &maybe_available_access_list);
    if (vm_stream->running_pkg_list().empty()) { active_vm_stream_list->Erase(vm_stream); }
  };
  auto* waiting_vm_instr_ctx_list = mut_waiting_vm_instr_ctx_list();
  ReadyVmInstrCtxList ready_vm_instr_ctx_list;
  if (waiting_msg_list().size() > 0) {
    TmpWaitingVmInstrMsgList tmp_waiting_msg_list;
    mut_waiting_msg_list()->MoveTo(&tmp_waiting_msg_list);
    FilterAndRunControlVmInstructions(this, &tmp_waiting_msg_list);
    auto* new_vm_instr_ctx_list = mut_new_vm_instr_ctx_list();
    MakeVmInstructionCtx(this, &tmp_waiting_msg_list, /*out*/ new_vm_instr_ctx_list);
    ConsumeMirroredObjects(mut_id2logical_object(), new_vm_instr_ctx_list,
                           /*out*/ &maybe_available_access_list);
    MoveToReadyCtxListIfNoObjectOperand(new_vm_instr_ctx_list, /*out*/ &ready_vm_instr_ctx_list);
    new_vm_instr_ctx_list->MoveTo(waiting_vm_instr_ctx_list);
  }
  FilterReadyVmInstrCtx(&maybe_available_access_list, waiting_vm_instr_ctx_list,
                        /*out*/ &ready_vm_instr_ctx_list);
  DispatchVmInstructionCtx(this, &ready_vm_instr_ctx_list);
}

}  // namespace oneflow
