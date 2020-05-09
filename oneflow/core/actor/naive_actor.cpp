#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

void NaiveActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void NaiveActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t piece_id = GetPieceId4NaiveCurReadableDataRegst();
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
}

REGISTER_ACTOR(TaskType::kReduceIdentity, NaiveActor);
REGISTER_ACTOR(TaskType::kSliceBoxing, NaiveActor);

}  // namespace oneflow
