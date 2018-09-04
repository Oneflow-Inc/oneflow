#include "oneflow/core/actor/copy_local_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyLocalActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyLocalActor::HandlerNormal);
}

void CopyLocalActor::Act() {
  Regst* in_regst = GetNaiveSoleCurReadable();
  AsyncLaunchKernel(GenDefaultKernelCtx());
  AsyncSendRegstMsgToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    out_regst->set_model_version_id(in_regst->model_version_id());
    return true;
  });
}

REGISTER_ACTOR(TaskType::kCopyLocal, CopyLocalActor);

#endif

}  // namespace oneflow
