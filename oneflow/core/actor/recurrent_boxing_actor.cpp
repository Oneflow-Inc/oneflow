#include "oneflow/core/actor/recurrent_boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void RecurrentBoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  ascending_status_ = 0;
  readable_regst_cnt_ = 0;
  is_eord_ = false;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
    previous_pst_[pair.second] = PieceStatus();
  }
  OF_SET_MSG_HANDLER(&RecurrentBoxingActor::HandlerNormal);
}

void RecurrentBoxingActor::TrySetAscendingStatus(const Regst* cur_regst) {
  PieceStatus& pre_pst = previous_pst_.at(cur_regst->regst_desc_id());
  const PieceStatus& cur_pst = cur_regst->piece_status();
  if (!pre_pst.max_col_num() || pre_pst.piece_id() != cur_pst.piece_id()) {
    pre_pst = cur_pst;
    return;
  }
  if (cur_pst.IsNextColOf(pre_pst)) {
    ascending_status_ = 1;
  } else {
    CHECK(pre_pst.IsNextColOf(cur_pst));
    ascending_status_ = -1;
  }
  return;
}

int RecurrentBoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      int64_t regst_desc_id = msg.regst()->regst_desc_id();
      if (!ascending_status_) { TrySetAscendingStatus(msg.regst()); }
      if (readable_regst_.at(regst_desc_id).empty()) {
        readable_regst_cnt_ += 1;
      }
      readable_regst_.at(regst_desc_id).push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void RecurrentBoxingActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regst_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    return regst->piece_status().col_id() < regst->piece_status().max_col_num();
  });
  int64_t cur_max_col_id = 0;
  int64_t cur_max_col_num = 0;
  for (const auto& pair : readable_regst_) {
    const PieceStatus& pst = pair.second.front()->piece_status();
    cur_max_col_id = std::max(cur_max_col_id, pst.col_id());
    cur_max_col_num = std::max(cur_max_col_num, pst.max_col_num());
  }
  for (auto& pair : readable_regst_) {
    const PieceStatus& pst = pair.second.front()->piece_status();
    if (ascending_status_ == 1) {
      if (pst.IsLastCol() && cur_max_col_id < cur_max_col_num - 1) { continue; }
    } else if (pst.col_id() < cur_max_col_id) {
      continue;
    }
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

bool RecurrentBoxingActor::IsReadReady() {
  CHECK_NE(0, ascending_status_);
  return readable_regst_.size() == readable_regst_cnt_;
}

bool RecurrentBoxingActor::IsReadAlwaysUnReadyFromNow() {
  return is_eord_ && !readable_regst_cnt_;
}

void RecurrentBoxingActor::AsyncReturnAllReadableRegst() {
  CHECK(readable_regst_.empty());
}

REGISTER_ACTOR(TaskType::kRecurrentBoxing, RecurrentBoxingActor);

}  // namespace oneflow
