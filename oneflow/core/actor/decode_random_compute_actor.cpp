#include "oneflow/core/actor/decode_random_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void DecodeRandomActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&DecodeRandomActor::HandlerNormal);
}

void DecodeRandomActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void DecodeRandomActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(kDecodeRandom, DecodeRandomActor);

}  // namespace oneflow
