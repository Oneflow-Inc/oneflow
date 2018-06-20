#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::Act(std::function<bool(Regst*)>* IsRegstAllowedSendActWiseMsgToConsumer) {
  Regst* in_regst = GetNaiveSoleCurReadable();
  AsyncLaunchKernel(GenDefaultKernelCtx());
  *IsRegstAllowedSendActWiseMsgToConsumer = [in_regst](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    out_regst->set_model_version_id(in_regst->model_version_id());
    return true;
  };
  AsyncSendRegstMsgToConsumer(*IsRegstAllowedSendActWiseMsgToConsumer);
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

#endif

}  // namespace oneflow
