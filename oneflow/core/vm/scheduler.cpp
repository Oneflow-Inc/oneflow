#include "oneflow/core/vm/scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

using VpuInstructionCtx = OBJECT_MSG_TYPE(VpuInstructionCtx);
using VpuSchedulerCtx = OBJECT_MSG_TYPE(VpuSchedulerCtx);
using VpuSetCtx = OBJECT_MSG_TYPE(VpuSetCtx);
using RunningVpuInstructionPackage = OBJECT_MSG_TYPE(RunningVpuInstructionPackage);
using WaitingVpuInstrCtxList = VpuSchedulerCtx::waiting_vpu_instr_ctx_list_ObjectMsgListType;
using ReadyVpuInstrCtxList = VpuSchedulerCtx::ready_vpu_instr_ctx_list_ObjectMsgListType;
using AvailableAccessList = VpuSchedulerCtx::available_access_list_ObjectMsgListType;
using TmpPendingVpuInstrMsgList = VpuSchedulerCtx::tmp_pending_msg_list_ObjectMsgListType;
using NewVpuInstrCtxList = VpuSchedulerCtx::new_vpu_instr_ctx_list_ObjectMsgListType;

namespace {

void ReleaseVpuInstructionCtx(VpuInstructionCtx* vpu_instr_ctx,
                              /* out */ AvailableAccessList* available_access_list) {
  auto* holding_operand_list = vpu_instr_ctx->mut_holding_operand_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(holding_operand_list, mirrored_object_access) {
    auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
    auto* holding_access_list = mirrored_object->mut_holding_access_list();
    holding_access_list->Erase(mirrored_object_access);
    if (!holding_access_list->empty()) { continue; }
    TODO();
  }
  holding_operand_list->Clear();
}

void ReleaseVpuInstructionPackage(RunningVpuInstructionPackage* pkg,
                                  /* out */ AvailableAccessList* available_access_list) {
  auto* vpu_instr_ctx_list = pkg->mut_vpu_instruction_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(vpu_instr_ctx_list, vpu_instr_ctx) {
    ReleaseVpuInstructionCtx(vpu_instr_ctx, /*out*/ available_access_list);
  }
  vpu_instr_ctx_list->Clear();
}

void TryReleaseFinishedVpuInstructionPackages(
    VpuSetCtx* vpu_set_ctx,
    /* out */ AvailableAccessList* available_access_list) {
  auto* pkg_list = vpu_set_ctx->mut_launched_pkg_list();
  while (true) {
    auto* begin = pkg_list->Begin();
    if (begin == nullptr || !begin->Done()) { break; }
    ReleaseVpuInstructionPackage(begin, /*out*/ available_access_list);
    pkg_list->Erase(begin);
  }
}

void FilterReadyVpuInstrCtx(AvailableAccessList* available_access_list,
                            WaitingVpuInstrCtxList* waiting_vpu_instr_ctx_list,
                            /* out */ ReadyVpuInstrCtxList* ready_vpu_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(available_access_list, mirrored_object_access) {
    auto* vpu_instruction_ctx = mirrored_object_access->mut_vpu_instruction_ctx();
    auto* waiting_operand_list = vpu_instruction_ctx->mut_waiting_operand_list();
    auto* holding_operand_list = vpu_instruction_ctx->mut_holding_operand_list();
    waiting_operand_list->MoveToDstBack(mirrored_object_access, holding_operand_list);
    if (waiting_operand_list->empty()) {
      waiting_vpu_instr_ctx_list->MoveToDstBack(vpu_instruction_ctx, ready_vpu_instr_ctx_list);
    }
  }
  available_access_list->Clear();
}

// fill new_ctx_list with VpuInstructionCtx which created by VpuInstructionMsg in
// tmp_pending_msg_list
void MakeVpuInstructionCtx(TmpPendingVpuInstrMsgList* vpu_instr_msg_list,
                           /* out */ NewVpuInstrCtxList* ret_vpu_instr_ctx_list) {
  TODO();
}

void ConsumeMirroredObjects(NewVpuInstrCtxList* new_vpu_instr_ctx_list,
                            /*out*/ AvailableAccessList* available_access_list) {
  TODO();
}

void DispatchVpuInstructionCtx(ReadyVpuInstrCtxList* ready_vpu_instr_ctx_list) { TODO(); }

}  // namespace

void VpuScheduler::Receive(VpuInstructionMsgList* vpu_instr_list) {
  std::unique_lock<std::mutex> lck(*ctx_->mut_pending_msg_list_mutex()->Mutable());
  vpu_instr_list->MoveTo(ctx_->mut_pending_msg_list());
}

void VpuScheduler::Dispatch() {
  auto* vpu_set_ctx_list = ctx_->mut_vpu_set_ctx_list();
  auto* available_access_list = ctx_->mut_available_access_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(vpu_set_ctx_list, vpu_set_ctx) {
    TryReleaseFinishedVpuInstructionPackages(vpu_set_ctx, /*out*/ available_access_list);
  };
  auto* waiting_vpu_instr_ctx_list = ctx_->mut_waiting_vpu_instr_ctx_list();
  auto* ready_vpu_instr_ctx_list = ctx_->mut_ready_vpu_instr_ctx_list();
  if (ctx_->pending_msg_list().size() > 0) {
    auto* tmp_pending_msg_list = ctx_->mut_tmp_pending_msg_list();
    {
      std::unique_lock<std::mutex> lck(*ctx_->mut_pending_msg_list_mutex()->Mutable());
      ctx_->mut_pending_msg_list()->MoveTo(tmp_pending_msg_list);
    }
    auto* new_vpu_instr_ctx_list = ctx_->mut_new_vpu_instr_ctx_list();
    MakeVpuInstructionCtx(tmp_pending_msg_list, /*out*/ new_vpu_instr_ctx_list);
    ConsumeMirroredObjects(new_vpu_instr_ctx_list, /*out*/ available_access_list);
  }
  FilterReadyVpuInstrCtx(available_access_list, waiting_vpu_instr_ctx_list,
                         /*out*/ ready_vpu_instr_ctx_list);
  DispatchVpuInstructionCtx(ready_vpu_instr_ctx_list);
}

}  // namespace oneflow
