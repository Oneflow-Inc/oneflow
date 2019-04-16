#include "oneflow/core/actor/every_nth_compute_actor.h"

namespace oneflow {

void EveryNthCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(exec_kernel_vec().size(), 1);
  every_nth_ = exec_kernel_vec().front().kernel->op_conf().every_nth_conf().n();
  current_nth_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&EveryNthCompActor::HandlerNormal);
}

void EveryNthCompActor::Act() {
  if (current_nth_ == every_nth_) { current_nth_ = 0; }
  current_nth_ += 1;
  if (current_nth_ == every_nth_) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    AsyncLaunchKernel(kernel_ctx);
  }
}

void EveryNthCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (current_nth_ == every_nth_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    cur_piece_id_ += 1;
  }
}

REGISTER_ACTOR(kEveryNth, EveryNthCompActor);

}  // namespace oneflow
