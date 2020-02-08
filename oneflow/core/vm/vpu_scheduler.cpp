#include "oneflow/core/vm/vpu_scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

using VpuSchedulerCtx = OBJECT_MSG_PTR(VpuSchedulerCtx);

namespace {}

VpuScheduler::VpuScheduler() { TODO(); }

void VpuScheduler::Receive(VpuInstructionMsgList* vpu_instr_list) {
  std::unique_lock<std::mutex> lck(*ctx_->mut_pending_list_mutex()->Mutable());
  vpu_instr_list->MoveTo(ctx_->mut_vpu_instruction_msg_pending_list());
}

void VpuScheduler::Schedule() {
  {
    std::unique_lock<std::mutex> lck(*ctx_->mut_pending_list_mutex()->Mutable());
    ctx_->mut_vpu_instruction_msg_pending_list()->MoveTo(ctx_->mut_tmp_pending_list());
  }
}

}  // namespace oneflow
