#include "oneflow/core/actor/recurrent_boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void RecurrentBoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  ascending_status_ = 0;
  readable_regst_cnt_ = 0;
  is_eord_ = false;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
    previous_pid_cid_[pair.second] = std::make_pair(-1, -1);
  }
  OF_SET_MSG_HANDLER(&RecurrentBoxingActor::HandlerNormal);
}

void RecurrentBoxingActor::TrySetAscendingStatus(const Regst* cur_regst) {
  auto& pre_pid_cid = previous_pid_cid_.at(cur_regst->regst_desc_id());
  int64_t cur_pid = cur_regst->piece_id();
  int64_t cur_cid = cur_regst->col_id();
  if (pre_pid_cid.first != cur_pid) {
    pre_pid_cid = std::make_pair(cur_pid, cur_cid);
    return;
  }
  if (cur_cid == pre_pid_cid.second + 1) {
    ascending_status_ = 1;
  } else {
    CHECK_EQ(cur_cid, pre_pid_cid - 1);
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
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { return regst->.col_id() <= regst->.max_col_id(); });
  int64_t cur_max_cid = 0;
  int64_t cur_max_maxcid = 0;
  for (const auto& pair : readable_regst_) {
    cur_max_cid = std::max(cur_max_cid, pair->col_id());
    cur_max_maxcid = std::max(cur_max_maxcid, pair->max_col_id());
  }
  for (auto& pair : readable_regst_) {
    if (ascending_status_ == 1) {
      if (pair->IsLastCol() && cur_max_cid < cur_max_maxcid) { continue; }
    } else if (pair->col_id() < cur_max_cid) {
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
