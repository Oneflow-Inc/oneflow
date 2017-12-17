#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(new CudaDeviceCtx(GetReservedWorkStreamId(0),
                                           thread_ctx.copy_hd_cuda_stream));
}

int CopyHdActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
    is_in_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      pending_in_regst_.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void CopyHdActor::Act() {
  Regst* in_regst = pending_in_regst_.front();
  pending_in_regst_.pop();
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](uint64_t regst_desc_id) -> Regst* {
                      if (regst_desc_id == in_regst->regst_desc_id()) {
                        return in_regst;
                      } else {
                        return GetCurWriteableRegst(regst_desc_id);
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    out_regst->set_model_version_id(in_regst->model_version_id());
    return true;
  });
  AsyncSendRegstMsgToProducer(in_regst);
}

bool CopyHdActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regst_.empty();
}

void CopyHdActor::AsyncReturnAllReadableRegst() {
  CHECK(pending_in_regst_.empty());
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

}  // namespace oneflow
