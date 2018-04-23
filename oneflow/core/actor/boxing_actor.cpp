#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  readable_regst_mgr_.Init(task_proto);
  previous_pid_cid_ = new HashMap<int64_t, std::pair<int64_t, int32_t>>;
  col_id_order_ = ColIdOrder::kUnCertain;
  is_eord_ = false;
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

void BoxingActor::TrySetColIdOrder(const Regst* cur_regst) {
  int64_t regst_desc_id = cur_regst->regst_desc_id();
  int64_t cur_pid = cur_regst->piece_id();
  int32_t cur_cid = cur_regst->col_id();
  if (previous_pid_cid_->find(regst_desc_id) == previous_pid_cid_->end()) {
    (*previous_pid_cid_)[regst_desc_id] = std::make_pair(cur_pid, cur_cid);
    return;
  }
  auto& pre_pid_cid = previous_pid_cid_->at(regst_desc_id);
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
  delete previous_pid_cid_;
  previous_pid_cid_ = nullptr;
  return;
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      if (msg.regst()->packed_blob()->max_col_num() > 1
          && col_id_order_ == ColIdOrder::kUnCertain) {
        TrySetColIdOrder(msg.regst());
      }
      readable_regst_mgr_.Push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void BoxingActor::Act() {
  int64_t piece_id = readable_regst_mgr_.GetFirstCurReadable()->piece_id();
  AsyncLaunchKernel(GenDefaultKernelCtx(), [this](int64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return readable_regst_mgr_.GetCurReadable(regst_desc_id);
    } else {
      return regst;
    }
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return regst->col_id() <= regst->max_col_id();
  });
  int32_t cur_max_cid = 0;
  int32_t cur_max_maxcid = 0;
  readable_regst_mgr_.ForEachCurReadableRegst([&](Regst* regst) {
    cur_max_cid = std::max(cur_max_cid, regst->col_id());
    cur_max_maxcid = std::max(cur_max_maxcid, regst->max_col_id());
  });
  readable_regst_mgr_.ReturnToProducerAndPopCurReadable(this, [&](Regst* regst) {
    if (col_id_order_ == ColIdOrder::kAscending) {
      if (regst->IsMaxCol() && cur_max_cid < cur_max_maxcid) { return false; }
    } else if (col_id_order_ == ColIdOrder::kDescending) {
      if (regst->col_id() < cur_max_cid) { return false; }
    } else {  // do nothing
    }
    return true;
  });
}

bool BoxingActor::IsReadReady() { return readable_regst_mgr_.IsReadReady(); }

bool BoxingActor::IsReadAlwaysUnReadyFromNow() { return is_eord_ && readable_regst_mgr_.IsEmpty(); }

void BoxingActor::AsyncReturnAllReadableRegst() { CHECK(readable_regst_mgr_.IsEmpty()); }

void BoxingActor::ForEachCurReadableRegst(std::function<void(const Regst*)> func) {
  readable_regst_mgr_.ForEachCurReadableRegst(func);
}

REGISTER_ACTOR(TaskType::kBoxing, BoxingActor);

}  // namespace oneflow
