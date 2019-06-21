#include "oneflow/core/actor/source_tick_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void SourceTickComputeActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&SourceTickComputeActor::HandlerWaitToStart);
}

void SourceTickComputeActor::Act() {
  Regst* regst = GetNaiveCurWriteable("out");
  regst->set_piece_id(piece_id_++);
}

bool SourceTickComputeActor::IsCustomizedReadReady() const {
  return piece_id_ < Global<RuntimeCtx>::Get()->total_piece_num();
}

int SourceTickComputeActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&SourceTickComputeActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kSourceTick, SourceTickComputeActor);

}  // namespace oneflow
