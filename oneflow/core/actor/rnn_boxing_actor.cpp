#include "oneflow/core/actor/rnn_boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void RnnBoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  num_of_consumed_ = task_proto.consumed_regst_desc_id().size();
  num_of_finished_in_cur_pid_ = 0;
  is_ascending_ = true;
  OF_SET_MSG_HANDLER(&RnnBoxingActor::HandlerNormal);
}

int RnnBoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      const PieceStatus& pst = msg.regst()->piece_status();
      int64_t cur_pid = pst.piece_id();
      int64_t regst_desc_id = msg.regst()->regst_desc_id();
      if (readable_regst_[cur_pid][regst_desc_id].empty()) {
        readable_regst_cnt_[cur_pid] += 1;
        if (pst.max_col_id() > 1 && pst.IsLastCol()) { is_ascending_ = false; }
      }
      readable_regst_[cur_pid][regst_desc_id].push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void RnnBoxingActor::Act() {
  auto& cur_readable_regst = readable_regst_.begin()->second;
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this, cur_readable_regst](int64_t regst_desc_id) -> Regst* {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          if (cur_readable_regst.at(regst_desc_id).empty()) {
            return nullptr;
          } else {
            return cur_readable_regst.at(regst_desc_id).front();
          }
        } else {
          return regst;
        }
      });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    return regst->piece_status().col_id() <= regst->piece_status().max_col_id();
  });
  for (auto& pair : cur_readable_regst) {
    const PieceStatus& pst = pair.second.front()->piece_status();
    if (pst.col_id() == pst.max_col_id()) { num_of_finished_in_cur_pid_ += 1; }
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
  if (num_of_finished_in_cur_pid_ == num_of_consumed_) {
    CHECK_EQ(0, readable_regst_cnt_);
    readable_regst_.erase(readable_regst_.begin());
    num_of_finished_in_cur_pid_ = 0;
  }
}

bool RnnBoxingActor::IsReadReady() {
  return readable_regst_cnt_ + num_of_finished_in_cur_pid_ == num_of_consumed_;
}

bool RnnBoxingActor::IsReadAlwaysUnReadyFromNow() {
  return !remaining_eord_cnt() && readable_regst_.empty();
}

void RnnBoxingActor::AsyncReturnAllReadableRegst() {
  CHECK(readable_regst_.empty());
}

REGISTER_ACTOR(TaskType::kRnnBoxing, RnnBoxingActor);

}  // namespace oneflow
