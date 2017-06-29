#include "oneflow/core/actor/model_diff_acc_actor.h"

namespace oneflow {

void MdDiffAccActor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  if (thread_ctx.cuda_stream) {
    clear_ek_.kernel = KernelMgr::Singleton().GetKernelFromOpName("gpu_clear");
    clear_ek_.bn_in_op2regst_desc_id = PbMap2HashMap();
  } else {
    clear_ek_.kernel = KernelMgr::Singleton().GetKernelFromOpName("cpu_clear");
    clear_ek_.bn_in_op2regst_desc_id = PbMap2HashMap();
  }
  OF_SET_MSG_HANDLE(&MdDiffAccActor::HandleMdDiffAcc);
}

int MdDiffAccActor::HandleMdDiffAcc(const ActorMsg&) {
}

int MdDiffAccActor::HandleMdDiffAccWhenNoReadableRegstMsg(const ActorMsg&) {
}

void MdDiffAccActor::TryWardKernelAndSendMsg() {
}

}  // namespace oneflow
