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

}  // namespace oneflow
