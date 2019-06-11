#include "oneflow/core/actor/tick_actor.h"

namespace oneflow {

void TickActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&TickActor::HandlerNormal);
}

void TickActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(piece_id_);
    return true;
  });
}

REGISTER_ACTOR(kTick, TickActor);

}  // namespace oneflow
