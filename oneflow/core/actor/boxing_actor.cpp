#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  num_of_subscribed_regsts_ = task_proto.subscribed_regst_desc_id().size();
  num_of_read_empty_ = num_of_subscribed_regsts_;
  num_of_read_done_ = 0;
  cur_msg_handle_ = &BoxingActor::HandleInitDeviceCtx;
}

int BoxingActor::ProcessMsg(const ActorMsg& msg,
                             const ThreadContext& thread_ctx) {
  return (this->*cur_msg_handle_)(msg, thread_ctx);
}

int BoxingActor::HandleInitDeviceCtx(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitDeviceCtx);
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  cur_msg_handle_ = &BoxingActor::HandleBoxing;
  return 0;
}

int BoxingActor::HandleBoxing(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kOneRegstDescDone);
    num_of_read_done_ += 1;
    if (num_of_read_done_ == num_of_subscribed_regsts_) {
      cur_msg_handle_ = &BoxingActor::HandleBoxingWhenNoReadableRegstMsg;
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()) != 0) {
      std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
      num_of_read_empty_ -= read_regst_[regst_wp->regst_desc_id()].empty();
      read_regst_.at(regst_wp->regst_desc_id()).push(regst_wp);
    } else {
      // do nothing
    }
  }
  TryWardKernelAndSendMsg();
  return 0;
}

int BoxingActor::HandleBoxingWhenNoReadableRegstMsg(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  TryWardKernelAndSendMsg();
  if (num_of_read_empty_ == num_of_subscribed_regsts_) {
    AsyncSendRegstDescDoneMsgForAllProducedRegstDesc();
    if (total_reading_cnt() == 0) {
      cur_msg_handle_ = nullptr;
      return 1;
    } else {
      cur_msg_handle_ = &BoxingActor::HandleWaitUntilReadingCntEqualZero;
      return 0;
    }
  }
  return 0;
}
  
int BoxingActor::HandleWaitUntilReadingCntEqualZero(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  if (total_reading_cnt() == 0) {
    cur_msg_handle_ = nullptr;
    return 1;
  }
  return 0;
}

void BoxingActor::TryWardKernelAndSendMsg() {
  if (!num_of_read_empty_ && IsWriteReady()) {
    uint64_t piece_id = expected_piece_id();
    for (const auto& pair : read_regst_) {
      CHECK_EQ(pair.second.front()->piece_id(), piece_id);
    }
    AsyncWardKernel(GenDefaultKernelCtx(), 
        [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        return read_regst_.at(regst_desc_id).front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([piece_id](Regst* regst) {
      regst->set_piece_id(piece_id);
    });
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
