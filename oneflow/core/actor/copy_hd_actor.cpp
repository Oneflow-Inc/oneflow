#include "oneflow/core/actor/copy_hd_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_wrapper.h"

namespace oneflow {

void CopyHdActor::Init(const TaskProto& task_proto,
                       const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(
      new CudaDeviceCtx(thread_ctx.copy_hd_cuda_stream, nullptr, nullptr));
  set_num_of_not_eord(1);
  mut_num_of_read_empty() = 1;
  OF_SET_MSG_HANDLE(&CopyHdActor::HandleNormal);
}

int CopyHdActor::HandleNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr())
        != 0) {
      mut_num_of_read_empty() = 0;
      waiting_in_regst_.push(msg.regst_wrapper());
    } else {
      // do nothing
    }
    ActUntilFail();
  }
  return msg_handle() == nullptr;
}

int CopyHdActor::HandleWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  ActUntilFail();
  if (mut_num_of_read_empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLE(&CopyHdActor::HandleWaitUntilReadingCntEqualZero);
  }
  return 0;
}

void CopyHdActor::Act() {
  std::shared_ptr<RegstWrapper> regst_wp = waiting_in_regst_.front();
  CHECK_EQ(regst_wp->piece_id(), expected_piece_id());
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
          return waiting_in_regst_.front();
        } else {
          return std::make_shared<LocalRegstWrapper>(regst);
        }
      });
  AsyncSendReadableRegstMsg([&regst_wp](Regst* regst) {
    regst->set_piece_id(regst_wp->piece_id());
    regst->set_model_version_id(regst_wp->model_version_id());
  });
  AsyncSendRegstMsgToProducer(regst_wp);
  waiting_in_regst_.pop();
  mut_num_of_read_empty() = waiting_in_regst_.empty();
}

REGISTER_ACTOR(kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
