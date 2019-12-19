#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  previous_pid_cid_.reset(new HashMap<int64_t, std::pair<int64_t, int32_t>>);
  col_id_order_ = ColIdOrder::kUnCertain;
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

void BoxingActor::NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>& rq) {
  if (rq.back()->regst_desc()->regst_desc_type().has_data_regst_desc() == false) { return; }
}

void BoxingActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void BoxingActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t piece_id = GetPieceId4NaiveCurReadableDataRegst();
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return regst->col_id() <= regst->max_col_id();
  });
}

void BoxingActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  int32_t cur_max_cid = 0;
  int32_t cur_max_maxcid = 0;
  ForEachCurNaiveReadableDataRegst([&](const Regst* regst) {
    cur_max_cid = std::max(cur_max_cid, regst->col_id());
    cur_max_maxcid = std::max(cur_max_maxcid, regst->max_col_id());
  });
  auto IsNaiveAllowedReturnToProducer = [this, cur_max_cid, cur_max_maxcid](Regst* regst) {
    if (col_id_order_ == ColIdOrder::kAscending) {
      if (regst->IsMaxCol() && cur_max_cid < cur_max_maxcid) { return false; }
    } else if (col_id_order_ == ColIdOrder::kDescending) {
      if (regst->col_id() < cur_max_cid) { return false; }
    } else {  // do nothing
    }
    return true;
  };
  HandleConsumedNaiveDataRegstToProducer(IsNaiveAllowedReturnToProducer);
}

void BoxingActor::TrySetColIdOrder(const Regst* regst) {
  int64_t regst_desc_id = regst->regst_desc_id();
  int64_t cur_pid = regst->piece_id();
  int32_t cur_cid = regst->col_id();
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
  previous_pid_cid_.reset();
  return;
}

REGISTER_ACTOR(TaskType::kBoxing, BoxingActor);

}  // namespace oneflow
