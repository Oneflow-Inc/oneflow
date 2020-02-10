#include "oneflow/core/vm/scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

using VpuInstructionCtx = OBJECT_MSG_TYPE(VpuInstructionCtx);
using VpuSchedulerCtx = OBJECT_MSG_TYPE(VpuSchedulerCtx);
using VpuSetCtx = OBJECT_MSG_TYPE(VpuSetCtx);
using RunningVpuInstructionPackage = OBJECT_MSG_TYPE(RunningVpuInstructionPackage);
using ReadyVpuInstrCtxList = VpuSchedulerCtx::ready_vpu_instr_ctx_list_ObjectMsgListType;
using ReleasedMirroredObjectList = VpuSchedulerCtx::released_mirrored_object_list_ObjectMsgListType;
using VpuInstrOperandAccessList =
    OBJECT_MSG_TYPE(MirroredObjectAccess)::vpu_instr_operand_list_ObjectMsgListType;

namespace {

void ReleaseVpuInstructionCtx(VpuInstructionCtx* vpu_instr_ctx,
                              /* out */ ReleasedMirroredObjectList* released_mirrored_object_list) {
  auto* holding_operand_list = vpu_instr_ctx->mut_holding_operand_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(holding_operand_list, vpu_instr_operand) {
    auto* mirrored_object_access = vpu_instr_operand->mut_mirrored_object_access();
    auto* vpu_instr_operand_list = mirrored_object_access->mut_vpu_instr_operand_list();
    vpu_instr_operand_list->Erase(vpu_instr_operand);
    if (!vpu_instr_operand_list->empty()) { continue; }
    auto* mirrored_object = vpu_instr_operand->mut_mirrored_object();
    auto* access_pending_list = mirrored_object->mut_access_pending_list();
    CHECK_EQ(access_pending_list->Begin(), mirrored_object_access);
    access_pending_list->Erase(mirrored_object_access);
    if (access_pending_list->Begin() == nullptr) { continue; }
    released_mirrored_object_list->PushBack(mirrored_object);
  }
  holding_operand_list->Clear();
}

void ReleaseVpuInstructionPackage(
    VpuSetCtx* ctx, RunningVpuInstructionPackage* pkg,
    /* out */ ReleasedMirroredObjectList* released_mirrored_object_list) {
  auto* vpu_instr_ctx_list = pkg->mut_vpu_instruction_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(vpu_instr_ctx_list, vpu_instr_ctx) {
    ReleaseVpuInstructionCtx(vpu_instr_ctx, released_mirrored_object_list);
  }
  vpu_instr_ctx_list->Clear();
}

void TryReleaseFinishedVpuInstructionPackages(
    VpuSetCtx* vpu_set_ctx,
    /* out */ ReleasedMirroredObjectList* released_mirrored_object_list) {
  auto* pkg_list = vpu_set_ctx->mut_launched_pkg_list();
  while (true) {
    auto* begin = pkg_list->Begin();
    if (begin == nullptr || !begin->Done()) { break; }
    ReleaseVpuInstructionPackage(vpu_set_ctx, begin, released_mirrored_object_list);
    pkg_list->Erase(begin);
  }
}

void ReleaseVpuInstrOperandAccesses(VpuInstrOperandAccessList* vpu_instr_operand_list,
                                    /* out */ ReadyVpuInstrCtxList* read_vpu_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(vpu_instr_operand_list, vpu_instr_operand) {
    auto* vpu_instruction_ctx = vpu_instr_operand->mut_vpu_instruction_ctx();
    auto* waiting_operand_list = vpu_instruction_ctx->mut_waiting_operand_list();
    auto* holding_operand_list = vpu_instruction_ctx->mut_holding_operand_list();
    waiting_operand_list->MoveToDstBack(vpu_instr_operand, holding_operand_list);
    if (waiting_operand_list->empty()) { read_vpu_instr_ctx_list->PushBack(vpu_instruction_ctx); }
  }
  vpu_instr_operand_list->Clear();
}

void FilterReadyVpuInstrCtx(ReleasedMirroredObjectList* released_mirrored_object_list,
                            /* out */ ReadyVpuInstrCtxList* read_vpu_instr_ctx_list) {
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(released_mirrored_object_list, mirrored_object) {
    auto* mirrored_object_access = mirrored_object->mut_access_pending_list()->Begin();
    if (mirrored_object_access == nullptr) { continue; }
    ReleaseVpuInstrOperandAccesses(mirrored_object_access->mut_vpu_instr_operand_list(),
                                   read_vpu_instr_ctx_list);
  }
  released_mirrored_object_list->Clear();
}

void MoveToTmpPendingMsgList(VpuSchedulerCtx* ctx) {
  std::unique_lock<std::mutex> lck(*ctx->mut_pending_msg_list_mutex()->Mutable());
  ctx->mut_pending_msg_list()->MoveTo(ctx->mut_tmp_pending_msg_list());
}

// fill new_ctx_list with VpuInstructionCtx which created by VpuInstructionMsg in
// tmp_pending_msg_list
void MakeVpuInstructionCtx(VpuSchedulerCtx* ctx) { TODO(); }

void MoveNewCtxToWaitingListOrReadyList(VpuSchedulerCtx* ctx) { TODO(); }

void DispatchVpuInstructionCtx(VpuSchedulerCtx* ctx) { TODO(); }

}  // namespace

void VpuScheduler::Receive(VpuInstructionMsgList* vpu_instr_list) {
  std::unique_lock<std::mutex> lck(*ctx_->mut_pending_msg_list_mutex()->Mutable());
  vpu_instr_list->MoveTo(ctx_->mut_pending_msg_list());
}

void VpuScheduler::Dispatch() {
  auto* vpu_set_ctx_list = ctx_->mut_vpu_set_ctx_list();
  auto* released_mirrored_object_list = ctx_->mut_released_mirrored_object_list();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(vpu_set_ctx_list, vpu_set_ctx) {
    TryReleaseFinishedVpuInstructionPackages(vpu_set_ctx, released_mirrored_object_list);
  };
  auto* ready_vpu_instr_ctx = ctx_->mut_ready_vpu_instr_ctx_list();
  FilterReadyVpuInstrCtx(released_mirrored_object_list, ready_vpu_instr_ctx);
  if (ctx_->pending_msg_list().size() > 0) {
    MoveToTmpPendingMsgList(ctx_);
    MakeVpuInstructionCtx(ctx_);
    MoveNewCtxToWaitingListOrReadyList(ctx_);
  }
  if (ready_vpu_instr_ctx->size() > 0) { DispatchVpuInstructionCtx(ctx_); }
}

}  // namespace oneflow
