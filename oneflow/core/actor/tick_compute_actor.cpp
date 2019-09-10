#include "oneflow/core/actor/tick_compute_actor.h"

namespace oneflow {

void TickComputeActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&TickComputeActor::HandlerNormal);
}

void TickComputeActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(piece_id_++);
    return true;
  });
}

REGISTER_ACTOR(kTick, TickComputeActor);

}  // namespace oneflow
