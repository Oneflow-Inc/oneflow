#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void CopyHdActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("copy_in");
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

#endif

}  // namespace oneflow
