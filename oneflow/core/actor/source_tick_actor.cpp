#include "oneflow/core/actor/source_tick_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void SourceTickActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&SourceTickActor::HandlerWaitToStart);
}

void SourceTickActor::Act() {
  Regst* regst = GetNaiveCurWriteable("out");
  regst->set_piece_id(piece_id_++);
}

bool SourceTickActor::IsCustomizedReadReady() {
  return piece_id_ < Global<RuntimeCtx>::Get()->total_piece_num();
}

int SourceTickActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&SourceTickActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kSourceTick, SourceTickActor);

}  // namespace oneflow
