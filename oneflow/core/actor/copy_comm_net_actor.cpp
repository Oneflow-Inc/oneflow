#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto,
                            const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLE(&CopyCommNetActor::HandleNormal);
}

int CopyCommNetActor::HandleNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    OF_SET_MSG_HANDLE(&CopyCommNetActor::HandleWaitUntilNoReadableRegst);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_wp = msg.regst_warpper();
    if (TryUpdtStateAsProducedRegst(regst_wp->regst_raw_ptr()) != 0) {
      CHECK(piece_id2waiting_in_regst_.emplace(regst_wp->piece_id(), regst_wp)
                .second);
    }
  }
  TryLaunchKernelAndSendMsg();
  return 0;
}

int CopyCommNetActor::HandleWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()),
           0);
  TryLaunchKernelAndSendMsg();
  if (piece_id2waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (total_reading_cnt() == 0) {
      OF_SET_MSG_HANDLE(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLE(&CopyCommNetActor::HandleWaitUntilReadingCntEqualZero);
      return 0;
    }
  }
  return 0;
}

void CopyCommNetActor::TryLaunchKernelAndSendMsg() {
  auto next_regst_it = piece_id2waiting_in_regst_.find(expected_piece_id());
  if (next_regst_it == piece_id2waiting_in_regst_.end()) { return; }
  if (IsWriteReady()) {
    std::shared_ptr<RegstWarpper> regst_wp = next_regst_it->second;
    AsyncLaunchKernel(
        GenDefaultKernelCtx(),
        [this,
         &regst_wp](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
          Regst* regst = GetCurWriteableRegst(regst_desc_id);
          if (regst == nullptr) {
            return regst_wp;
          } else {
            return std::make_shared<LocalRegstWarpper>(regst);
          }
        });
    ForEachCurWriteableRegst([&regst_wp](Regst* regst) {
      regst->set_piece_id(regst_wp->piece_id());
      regst->set_model_version_id(regst_wp->model_version_id());
    });
    AsyncSendReadableRegstMsg();
    AsyncSendRegstMsgToProducer(regst_wp);
    piece_id2waiting_in_regst_.erase(next_regst_it);
  }
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
