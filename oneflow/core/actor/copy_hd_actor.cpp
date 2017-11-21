#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  set_num_of_remaining_eord(1);
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(new CudaDeviceCtx(thread_ctx.copy_hd_cuda_stream));
}

int CopyHdActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      pending_in_regst_.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int CopyHdActor::HandlerUntilReadAlwaysUnReady(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (IsReadAlwaysUnReadyFromNow()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&CopyHdActor::HandlerZombie);
  }
  return 0;
}

bool CopyHdActor::IsReadAlwaysUnReadyFromNow() {
  return pending_in_regst_.empty();
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
  });
  AsyncSendRegstMsgToProducer(in_regst);
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

}  // namespace oneflow
