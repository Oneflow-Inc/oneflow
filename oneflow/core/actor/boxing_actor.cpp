#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
  }
  readable_regst_cnt_ = 0;
  col_id_order_ = ColIdOrder::kUncertain;
  is_eord_ = false;
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

void BoxingActor::TrySetColIdOrder(const Regst* cur_regst) {
  int64_t regst_desc_id = cur_regst->regst_desc_id();
  int64_t cur_pid = cur_regst->piece_id();
  int64_t cur_cid = cur_regst->col_id();
  if (previous_pid_cid_.find(regst_desc_id) == previous_pid_cid_.end()) {
    previous_pid_cid_[regst_desc_id] = std::make_pair(cur_pid, cur_cid);
  }
  auto& pre_pid_cid = previous_pid_cid_.at(regst_desc_id);
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
  previous_pid_cid_.clear();
  return;
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      if (col_id_order_ == ColIdOrder::kUncertain) {
        TrySetColIdOrder(msg.regst());
      }
      std::queue<Regst*>& rq = readable_regst_.at(msg.regst()->regst_desc_id());
      if (rq.empty()) { readable_regst_cnt_ += 1; }
      rq.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void BoxingActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  std::function<void(const std::string&, int64_t)> fn = 
    [this](const std::string& bn, int32_t max_col) {
      int regst_desc_id = exec_kernel_vec()[0].bn_in_op2regst_desc_id.at(bn);
    };
  //kernel_ctx.other = &data_load_status_;
  AsyncLaunchKernel(kernel_ctx,
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regst_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(regst->piece_id());
    return regst->col_id() <= regst->max_col_id();
  });
  int64_t cur_max_cid = 0;
  int64_t cur_max_maxcid = 0;
  for (const auto& pair : readable_regst_) {
    cur_max_cid = std::max(cur_max_cid, pair.second.front()->col_id());
    cur_max_maxcid =
        std::max(cur_max_maxcid, pair.second.front()->max_col_id());
  }
  for (auto& pair : readable_regst_) {
    if (col_id_order_ == ColIdOrder::kAscending) {
      if (pair.second.front()->IsMaxCol() && cur_max_cid < cur_max_maxcid) {
        continue;
      }
    } else if (col_id_order_ == ColIdOrder::kDescending) {
      if (pair.second.front()->col_id() < cur_max_cid) { continue; }
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
