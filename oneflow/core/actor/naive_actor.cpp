#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

void NaiveActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx());
  int64_t piece_id = GetNaiveFirstCurReadable()->piece_id();
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
}

REGISTER_ACTOR(TaskType::kReduceScatter, NaiveActor);
REGISTER_ACTOR(TaskType::kReduceAdd, NaiveActor);
REGISTER_ACTOR(TaskType::kReduceGather, NaiveActor);

}  // namespace oneflow
