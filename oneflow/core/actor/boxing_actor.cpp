#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
    previous_pid_cid_[pair.second] = std::make_pair(-1, -1);
  }
  readable_regst_cnt_ = 0;
  col_id_order_ = ColIdOrder::kUnset;
  is_eord_ = false;
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

void BoxingActor::TrySetAscendingStatus(const Regst* cur_regst) {
  auto& pre_pid_cid = previous_pid_cid_.at(cur_regst->regst_desc_id());
  int64_t cur_pid = cur_regst->piece_id();
  int64_t cur_cid = cur_regst->col_id();
  if (pre_pid_cid.first != cur_pid) {
    pre_pid_cid = std::make_pair(cur_pid, cur_cid);
    return;
  }
  if (cur_cid == pre_pid_cid.second + 1) {
    col_id_order_ = ColIdOrder::kAscending;
  } else {
    CHECK_EQ(cur_cid, pre_pid_cid.second - 1);
    col_id_order_ = ColIdOrder::kDescending;
  }
  return;
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      int64_t regst_desc_id = msg.regst()->regst_desc_id();
      if (col_id_order_ == ColIdOrder::kUnset) {
        TrySetAscendingStatus(msg.regst());
      }
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

void BoxingActor::Act() {
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
      [&](Regst* regst) { return regst->col_id() <= regst->max_col_id(); });
  int64_t cur_max_cid = 0;
  int64_t cur_max_maxcid = 0;
  for (const auto& pair : readable_regst_) {
    cur_max_cid = std::max(cur_max_cid, pair.second.front()->col_id());
    cur_max_maxcid =
        std::max(cur_max_maxcid, pair.second.front()->max_col_id());
  }
  for (auto& pair : readable_regst_) {
    if (col_id_order_ == ColIdOrder::kAscending) {
      if (pair.second.front()->IsLastCol() && cur_max_cid < cur_max_maxcid) {
        continue;
      }
    } else if (pair.second.front()->col_id() < cur_max_cid) {
      continue;
    } else {  // do nothing
    }
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

bool BoxingActor::IsReadReady() {
  return readable_regst_.size() == readable_regst_cnt_;
}

bool BoxingActor::IsReadAlwaysUnReadyFromNow() {
  return is_eord_ && readable_regst_cnt_ == 0;
}

void BoxingActor::AsyncReturnAllReadableRegst() {
  CHECK_EQ(readable_regst_cnt_, 0);
}

REGISTER_ACTOR(TaskType::kBoxing, BoxingActor);

}  // namespace oneflow
