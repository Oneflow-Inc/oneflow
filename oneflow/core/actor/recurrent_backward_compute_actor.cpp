#include "oneflow/core/actor/recurrent_backward_compute_actor.h"

namespace oneflow {

void RecurrentBackwardCompActor::VirtualBackwardCompActorInit(
    const TaskProto& task_proto) {
  h0_regst_desc_id_ = RegstDescId4Name("h0");
  rec_out_diff_regst_desc_id_ = RegstDescId4Name("rec_acc_diff");
  if (rec_out_diff_regst_desc_id_ == -1) {
    CHECK(parallel_ctx()->policy() == kDataParallel);
  } else {
    CHECK(parallel_ctx()->policy() == kModelParallel);
  }

  rec_out_diff_regst_ = nullptr;
  model_tmp_regst_ = nullptr;
  order_ = ColIdOrder::kUnCertain;
  OF_SET_MSG_HANDLER(&RecurrentBackwardCompActor::HandlerNormal);
}

void RecurrentBackwardCompActor::TryUpdtColIdOrder(const Regst* cur_regst) {
  if (order_ == ColIdOrder::kUnCertain) {
    if (cur_regst->col_id() == 0) {
      if (!(cur_regst->IsMaxCol())) {
        order_ = ColIdOrder::kAscending;
      } else {
        CHECK_EQ(0, cur_regst->max_col_id());
      }
    } else if (cur_regst->IsMaxCol()) {
      order_ = ColIdOrder::kDescending;
    } else {
      UNEXPECTED_RUN();
    }
  }
}

int RecurrentBackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id()) {
      set_is_out_diff_eord(true);
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      int64_t cur_col_id = cur_regst->col_id();

      if (cur_regst_desc_id == in_regst_desc_id()) {
        if (cur_col_id == 0) { in_regsts_.push_back(std::stack<Regst*>()); }
        in_regsts_.back().push(cur_regst);
      } else if (cur_regst_desc_id == out_regst_desc_id()) {
        if (cur_col_id == 0) {
          out_regsts_.push_back(std::deque<Regst*>{cur_regst});
          if (out_regsts_.size() == 1) {
            AsyncReturnModelRegstUntilMatchCurOutRegst(
                cur_regst->model_version_id(), model_regsts_);
          }
        } else {
          out_regsts_.back().push_back(cur_regst);
        }
      } else if (cur_regst_desc_id == h0_regst_desc_id_) {
        h0_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == out_diff_regst_desc_id()) {
        TryUpdtColIdOrder(cur_regst);
        if ((order_ == ColIdOrder::kUnCertain)
            || IsFirstRegstInPieceWithOrder(cur_regst, order_)) {
          out_diff_regsts_.push_back(std::deque<Regst*>());
        }
        if (order_ == ColIdOrder::kUnCertain) {
          CHECK_EQ(0, cur_regst->max_col_id());
          out_diff_regsts_.back().push_back(cur_regst);
        } else if (order_ == ColIdOrder::kAscending) {
          out_diff_regsts_.back().push_back(cur_regst);
        } else {
          out_diff_regsts_.back().push_front(cur_regst);
        }
      } else if (cur_regst_desc_id == rec_out_diff_regst_desc_id_) {
        CHECK(!rec_out_diff_regst_);
        if (cur_col_id == 0) {
          AsyncSendRegstMsgToProducer(cur_regst);
        } else {
          rec_out_diff_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == model_regst_desc_id()) {
        model_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_tmp_regst_desc_id()) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = cur_regst;
      } else if (cur_regst_desc_id == data_tmp_regst_desc_id()) {
        if (cur_col_id == 0) {
          data_tmp_regsts_.push_back(std::stack<Regst*>());
        }
        data_tmp_regsts_.back().push(cur_regst);
      } else {
        UNEXPECTED_RUN();
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool RecurrentBackwardCompActor::IsReadReady() {
  if (in_regsts_.empty() || out_regsts_.empty() || out_diff_regsts_.empty()
      || model_regsts_.empty() || data_tmp_regsts_.empty()) {
    return false;
  }
  if (model_tmp_regst_desc_id() != -1 && !model_tmp_regst_) { return false; }

  Regst* out_regst = out_regsts_.front().back();
  if (out_regst->col_id() == 0) {
    if (h0_regst_desc_id_ != -1 && h0_regsts_.empty()) { return false; }
  }

  if (out_regst->IsMaxCol()) {
    CHECK(!rec_out_diff_regst_);
    if (!(in_regsts_.front().top()->IsMaxCol())) { return false; }
    if (!(out_diff_regsts_.front().back()->IsMaxCol())) { return false; }
    if (!(data_tmp_regsts_.front().top()->IsMaxCol())) { return false; }
  } else {
    if (!rec_out_diff_regst_) { return false; }
    CHECK(rec_out_diff_regst_->HaveNextPieceColStatusOf(out_regst));
    CHECK(in_regsts_.front().top()->HaveSamePieceColStatusAs(out_regst));
    CHECK(out_diff_regsts_.front().back()->HaveSamePieceColStatusAs(out_regst));
    CHECK(data_tmp_regsts_.front().top()->HaveSamePieceColStatusAs(out_regst));
  }
  return true;
}

void RecurrentBackwardCompActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      if (regst_desc_id == in_regst_desc_id()) {
                        return in_regsts_.front().top();
                      } else if (regst_desc_id == out_regst_desc_id()) {
                        return out_regsts_.front().back();
                      } else if (regst_desc_id == data_tmp_regst_desc_id()) {
                        return data_tmp_regsts_.front().top();
                      } else if (regst_desc_id == h0_regst_desc_id_) {
                        return h0_regsts_.front();
                      } else if (regst_desc_id == out_diff_regst_desc_id()) {
                        return out_diff_regsts_.front().back();
                      } else if (regst_desc_id == rec_out_diff_regst_desc_id_) {
                        return rec_out_diff_regst_;
                      } else if (regst_desc_id == model_regst_desc_id()) {
                        return model_regsts_.front();
                      } else if (regst_desc_id == model_tmp_regst_desc_id()) {
                        return model_tmp_regst_;
                      } else {
                        return GetCurWriteableRegst(regst_desc_id);
                      }
                    });
  AsyncSendRegstMsgToConsumer([this](Regst* regst) {
    if (parallel_ctx()->policy() == kDataParallel
        && regst->regst_desc_id() == RegstDescId4Name("rec_out_diff")) {
      return false;
    }
    regst->set_piece_id(out_diff_regsts_.front().back()->piece_id());
    regst->set_piece_id(out_diff_regsts_.front().back()->col_id());
    regst->set_piece_id(out_diff_regsts_.front().back()->max_col_id());
    return true;
  });

  Regst* out_regst = out_regsts_.front().back();
  out_regsts_.front().pop_back();

  AsyncSendRegstMsgToProducer(in_regsts_.front().top());
  in_regsts_.front().pop();
  AsyncSendRegstMsgToProducer(out_diff_regsts_.front().back());
  out_diff_regsts_.front().pop_back();
  AsyncSendRegstMsgToProducer(data_tmp_regsts_.front().top());
  data_tmp_regsts_.front().pop();

  if (!(out_regst->IsMaxCol())) {
    AsyncSendRegstMsgToProducer(rec_out_diff_regst_);
    rec_out_diff_regst_ = nullptr;
  }
  if (out_regst->col_id() == 0) {
    AsyncSendRegstMsgToProducer(h0_regsts_.front());
    h0_regsts_.pop();

    CHECK(in_regsts_.front().empty());
    in_regsts_.pop_front();
    CHECK(out_diff_regsts_.front().empty());
    out_diff_regsts_.pop_front();
    CHECK(data_tmp_regsts_.front().empty());
    data_tmp_regsts_.pop_front();
    CHECK(out_regsts_.front().empty());
    out_regsts_.pop_front();

    if (out_regsts_.empty()) {
      AsyncReturnModelRegstUntilLastPieceIdGreaterThan(out_regst->piece_id(),
                                                       model_regsts_);
    } else {
      AsyncReturnModelRegstUntilMatchCurOutRegst(out_regst->model_version_id(),
                                                 model_regsts_);
    }
  }
  AsyncSendRegstMsgToProducer(out_regst);
}

bool RecurrentBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord() && out_diff_regsts_.empty();
}

void RecurrentBackwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(in_regsts_.empty());
  CHECK(out_regsts_.empty());
  CHECK(data_tmp_regsts_.empty());
  CHECK(out_diff_regsts_.empty());
  CHECK(h0_regsts_.empty());
  CHECK(!rec_out_diff_regst_);

  if (model_tmp_regst_) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }

  while (!model_regsts_.empty()) {
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
}

REGISTER_ACTOR(TaskType::kRecurrentBackward, RecurrentBackwardCompActor);

}  // namespace oneflow
