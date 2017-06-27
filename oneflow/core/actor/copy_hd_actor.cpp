#include "oneflow/core/actor/copy_hd_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyHdActor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(new CudaDeviceCtx(thread_ctx.copy_hd_cuda_stream,
                                           nullptr,
                                           nullptr));
  OF_SET_MSG_HANDLE(&CopyHdActor::HandleCopyHd);
}

int CopyHdActor::HandleCopyHd(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    OF_SET_MSG_HANDLE(&CopyHdActor::HandleCopyHdWhenNoReadableRegstMsg);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()) != 0) {
      waiting_in_regst_.push(msg.regst_warpper());
    }
  }
  TryWardKernelAndSendMsg();
  return 0;
}

int CopyHdActor::HandleCopyHdWhenNoReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  TryWardKernelAndSendMsg();
  if (waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (total_reading_cnt() == 0) {
      OF_SET_MSG_HANDLE(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLE(&CopyHdActor::HandleWaitUntilReadingCntEqualZero);
      return 0;
    }
  }
  return 0;
}

void CopyHdActor::TryWardKernelAndSendMsg() {
  if (!waiting_in_regst_.empty() && IsWriteReady()) {
    std::shared_ptr<RegstWarpper> regst_wp = waiting_in_regst_.front();
    CHECK_EQ(regst_wp->piece_id(), expected_piece_id());
    AsyncWardKernel(GenDefaultKernelCtx(),
        [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
        return waiting_in_regst_.front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([&regst_wp](Regst* regst) {
      regst->set_piece_id(regst_wp->piece_id());
      regst->set_model_version_id(regst_wp->model_version_id());
    });
    AsyncSendReadableRegstMsg();
    AsyncSendRegstMsgToProducer(regst_wp);
    waiting_in_regst_.pop();
  }
}

REGISTER_ACTOR(kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
