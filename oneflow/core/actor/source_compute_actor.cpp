#include "oneflow/core/actor/source_compute_actor.h"

namespace oneflow {

void SourceCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  next_piece_id_ = 0;
  is_eof_ = false;
  OF_SET_MSG_HANDLER(&SourceCompActor::HandlerWaitToStart);
}

int SourceCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLER(&SourceCompActor::HandlerNormal);
  return 0;
}

int SourceCompActor::HandlerNormal(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  return TrySwitchToZombieOrFinish();
}

void SourceCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &is_eof_;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  AsyncSendRegstMsgToConsumer([this](Regst* regst) {
    regst->set_piece_id(next_piece_id_);
    return true;
  });
  next_piece_id_ += 1;
}

bool SourceCompActor::IsReadReady() {
  return is_eof_ == false
         && next_piece_id_ < RuntimeCtx::Singleton()->total_piece_num();
}

std::list<RegstEvent> SourceCompActor::CurActComsumedRegstEvents() const {
  TODO();
  return {};
}

REGISTER_ACTOR(kSource, SourceCompActor);

}  // namespace oneflow
