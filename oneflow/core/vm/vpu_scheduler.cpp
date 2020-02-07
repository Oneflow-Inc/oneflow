#include "oneflow/core/vm/vpu_scheduler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

VpuScheduler::VpuScheduler() { TODO(); }

void VpuScheduler::Receive(OBJECT_MSG_PTR(VpuInstructionMsg) && vpu_instr_msg) {
  std::unique_lock<std::mutex> lck(mutex_);
  ctx_->mutable_vpu_instruction_msg_pending_list()->PushBack(vpu_instr_msg.Mutable());
}

void VpuScheduler::MainLoop() {
  while (true) { Schedule(); }
}

void VpuScheduler::Schedule() {
  {
    std::unique_lock<std::mutex> lck(mutex_);
    ctx_->mutable_vpu_instruction_msg_pending_list()->MoveTo(ctx_->mutable_tmp_list());
  }
}

}  // namespace oneflow
