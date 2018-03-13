#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

void BackwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  out_regst_desc_id_ = RegstDescId4Name("out");
  in_regst_desc_id_ = RegstDescId4Name("in");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");

  is_out_diff_eord_ = false;
  order_ = ColIdOrder::kUnCertain;
  model_tmp_regst_ = nullptr;
  has_cur_piece_started_ = false;

  VirtualBackwardCompActorInit(task_proto);
}

int BackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id_) {
      is_out_diff_eord_ = true;
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      if (cur_regst_desc_id == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = cur_regst;
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        model_regsts_.push(cur_regst);
      } else {
        HandleTheRestOfRegstMsg(cur_regst);
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void BackwardCompActor::HandleOutDiffRegsts(
    Regst* cur_regst, std::deque<std::deque<Regst*>>* out_diff_regsts) {
  TryUpdtColIdOrder(cur_regst, &order_);
  if ((order() == ColIdOrder::kUnCertain)
      || IsFirstRegstInPieceWithOrder(cur_regst, order())) {
    out_diff_regsts->push_back(std::deque<Regst*>());
  }
  if (order() == ColIdOrder::kUnCertain) {
    CHECK_EQ(0, cur_regst->max_col_id());
    out_diff_regsts->back().push_back(cur_regst);
  } else if (order() == ColIdOrder::kAscending) {
    out_diff_regsts->back().push_back(cur_regst);
  } else {
    out_diff_regsts->back().push_front(cur_regst);
  }
}

void BackwardCompActor::AsyncReturnModelRegstUntilMatchCurOutRegst(
    int64_t cur_model_id) {
  while (!model_regsts_.empty()
         && model_regsts_.front()->model_version_id() < cur_model_id) {
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
  if (!model_regsts_.empty()) {
    CHECK_EQ(model_regsts_.front()->model_version_id(), cur_model_id);
  }
}

void BackwardCompActor::AsyncReturnModelRegstUntilLastPieceIdGreaterThan(
    int64_t piece_id) {
  while (model_regsts_.empty() == false) {
    int64_t model_id = model_regsts_.front()->model_version_id();
    int64_t last_piece_id = GetLastPieceIdForModelVersionId(model_id);
    if (last_piece_id > piece_id) { return; }
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
}

void BackwardCompActor::AsyncReturnAllReadableRegst() {
  CheckBeforeAsyncReturnAllReadableRegst();
  TryAsyncReturnModelRegst();
  TryAsyncReturnModelTmpRegst();
}

void BackwardCompActor::TryAsyncReturnModelRegst() {
  while (!model_regsts_.empty()) {
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
}

void BackwardCompActor::TryAsyncReturnModelTmpRegst() {
  if (model_tmp_regst_) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
}

void BackwardCompActor::ForCurReadableModelAndModelTmp(
    std::function<void(const Regst*)> handler) {
  if (model_regst_desc_id_ != -1) { handler(model_regsts_.front()); }
  if (model_tmp_regst_desc_id_ != -1) { handler(model_tmp_regst_); }
}

bool BackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_
         && readable_deq_regsts_.at(out_diff_regst_desc_id_).empty();
}

}  // namespace oneflow
