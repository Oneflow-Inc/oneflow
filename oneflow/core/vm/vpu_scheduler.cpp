#include "oneflow/core/vm/vpu_scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

VpuScheduler::VpuScheduler() { TODO(); }

void VpuScheduler::Receive(OBJECT_MSG_PTR(VpuInstructionMsg) && vpu_instr_msg) {
  std::unique_lock<std::mutex> lck(*ctx_->mut_pending_list_mutex()->Mutable());
  ctx_->mut_vpu_instruction_msg_pending_list()->PushBack(vpu_instr_msg.Mutable());
}

void VpuScheduler::MainLoop() {
  while (true) { Schedule(); }
}

void VpuScheduler::Schedule() {
  {
    std::unique_lock<std::mutex> lck(*ctx_->mut_pending_list_mutex()->Mutable());
    ctx_->mut_vpu_instruction_msg_pending_list()->MoveTo(ctx_->mut_tmp_pending_list());
  }
}

}  // namespace oneflow
