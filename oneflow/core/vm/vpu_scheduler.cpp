#include "oneflow/core/vm/vpu_scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

using VpuSchedulerCtx = OBJECT_MSG_TYPE(VpuSchedulerCtx);
using VpuSetCtx = OBJECT_MSG_TYPE(VpuSetCtx);
using RunningVpuInstructionPackage = OBJECT_MSG_TYPE(RunningVpuInstructionPackage);

namespace {

void HandleFinishedVpuInstructionCtx(VpuSchedulerCtx* ctx, RunningVpuInstructionPackage* pkg) {
  TODO();
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
  auto* ctx = ctx_.Mutable();
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(ctx->mut_vpu_set_ctx_list(), vpu_set_ctx) {
    auto* begin = vpu_set_ctx->mut_launched_pkg_list()->Begin();
    if (begin == nullptr || !begin->Done()) { continue; }
    HandleFinishedVpuInstructionCtx(ctx, begin);
  };
  if (ctx->pending_msg_list().size() > 0) {
    MoveToTmpPendingMsgList(ctx);
    MakeVpuInstructionCtx(ctx);
    MoveNewCtxToWaitingListOrReadyList(ctx);
  }
  if (ctx->ready_ctx_list().size() > 0) { DispatchVpuInstructionCtx(ctx); }
}

}  // namespace oneflow
