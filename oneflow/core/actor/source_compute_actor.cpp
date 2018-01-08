#include "oneflow/core/actor/source_compute_actor.h"

namespace oneflow {

void SourceCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
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
  kernel_ctx.other = &data_load_status;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  AsyncSendRegstMsgToConsumer([this](Regst* regst) {
    regst->set_piece_id(data_load_status.piece_id);
    return true;
  });
}

bool SourceCompActor::IsReadReady() {
  return data_load_status.next_col_id != data_load_status.max_col_num
         || (data_load_status.is_eof == false
             && data_load_status.piece_id
                    < RuntimeCtx::Singleton()->total_piece_num());
}

REGISTER_ACTOR(kSource, SourceCompActor);

}  // namespace oneflow
