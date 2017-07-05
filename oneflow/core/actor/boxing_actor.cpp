#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto,
                       const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  num_of_subscribed_regsts_ = task_proto.subscribed_regst_desc_id().size();
  num_of_read_empty_ = num_of_subscribed_regsts_;
  num_of_eord_ = 0;
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLE(&BoxingActor::HandleNormal);
}

int BoxingActor::HandleNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    num_of_eord_ += 1;
    if (num_of_eord_ == num_of_subscribed_regsts_) {
      OF_SET_MSG_HANDLE(&BoxingActor::HandleWaitUntilNoReadableRegst);
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr())
        != 0) {
      std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
      num_of_read_empty_ -= read_regst_[regst_wp->regst_desc_id()].empty();
      read_regst_.at(regst_wp->regst_desc_id()).push(regst_wp);
    } else {
      // do nothing
    }
  }
  TryLaunchKernelAndSendMsg();
  return 0;
}

int BoxingActor::HandleWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()),
           0);
  TryLaunchKernelAndSendMsg();
  if (num_of_read_empty_ == num_of_subscribed_regsts_) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (total_reading_cnt() == 0) {
      OF_SET_MSG_HANDLE(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLE(&BoxingActor::HandleWaitUntilReadingCntEqualZero);
      return 0;
    }
  }
  return 0;
}

void BoxingActor::TryLaunchKernelAndSendMsg() {
  if (!num_of_read_empty_ && IsWriteReady()) {
    int64_t piece_id = expected_piece_id();
    for (const auto& pair : read_regst_) {
      CHECK_EQ(pair.second.front()->piece_id(), piece_id);
    }
    AsyncLaunchKernel(
        GenDefaultKernelCtx(),
        [this](int64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
          Regst* regst = GetCurWriteableRegst(regst_desc_id);
          if (regst == nullptr) {
            return read_regst_.at(regst_desc_id).front();
          } else {
            return std::make_shared<LocalRegstWarpper>(regst);
          }
        });
    ForEachCurWriteableRegst(
        [piece_id](Regst* regst) { regst->set_piece_id(piece_id); });
    AsyncSendReadableRegstMsg();
    for (auto& pair : read_regst_) {
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop();
      num_of_read_empty_ += pair.second.empty();
    }
  }
}

REGISTER_ACTOR(kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(kBoxingTask, false, BoxingActor);

}  // namespace oneflow
