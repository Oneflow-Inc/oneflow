#include "oneflow/core/actor/decode_random_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void DecodeRandomActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&DecodeRandomActor::HandlerWaitToStart);
}

void DecodeRandomActor::Act() {
  Regst* regst = GetNaiveCurWriteable("out");
  regst->set_piece_id(piece_id_++);

  AsyncLaunchKernel(GenDefaultKernelCtx());
}

bool DecodeRandomActor::IsCustomizedReadReady() const {
  return piece_id_ < Global<RuntimeCtx>::Get()->total_piece_num();
}

int DecodeRandomActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&DecodeRandomActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kDecodeRandom, DecodeRandomActor);

}  // namespace oneflow
