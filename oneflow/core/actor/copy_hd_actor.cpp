#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto,
                                   const ThreadCtx& thread_ctx) {
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(
      new CudaDeviceCtx(thread_ctx.copy_hd_cuda_stream, nullptr, nullptr));
  set_num_of_remaining_eord(1);
  mut_num_of_read_empty() = 1;
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

int CopyHdActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      mut_num_of_read_empty() = 0;
      waiting_in_regst_.push(msg.regst());
    } else {
      // do nothing
    }
    ActUntilFail();
  }
  return msg_handler() == nullptr;
}

int CopyHdActor::HandlerUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (num_of_read_empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&CopyHdActor::HandlerZombie);
  }
  return 0;
}

void CopyHdActor::Act() {
  Regst* in_regst = waiting_in_regst_.front();
  // CopyHdActor in model update path does not set piece_id, ommit the CHECK
  // CHECK_EQ(in_regst->piece_id(), expected_piece_id());
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](uint64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        CHECK_EQ(regst_desc_id, in_regst->regst_desc_id());
                        return in_regst;
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    out_regst->set_model_version_id(in_regst->model_version_id());
  });
  AsyncSendRegstMsgToProducer(in_regst);
  waiting_in_regst_.pop();
  mut_num_of_read_empty() = waiting_in_regst_.empty();
}

}  // namespace oneflow
