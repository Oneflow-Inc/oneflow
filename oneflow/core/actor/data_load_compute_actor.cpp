#include "oneflow/core/actor/data_load_compute_actor.h"

namespace oneflow {

void DataLoadActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void DataLoadActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(kDataLoad, DataLoadActor);

}  // namespace oneflow
