#include "oneflow/core/actor/recurrent_backward_compute_actor.h"

namespace oneflow {

void RecurrentBackwardCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  in_regst_desc_id_ = RegstDescId4Name("in");
  out_regst_desc_id_ = RegstDescId4Name("out");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  rec_acc_diff_regst_desc_id_ = RegstDescId4Name("rec_acc_diff");
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");

  is_out_diff_eord_ = false;
  rec_acc_diff_regst_ = nullptr;
  model_tmp_regst_ = nullptr;
  insert_order_ = 0;
  OF_SET_MSG_HANDLER(&RecurrentBackwardCompActor::HandlerNormal);
}

int RecurrentBackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id_) {
      is_out_diff_eord_ = true;
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      const PieceStatus& cur_pst = cur_regst->piece_status();
      int64_t cur_col_id = cur_pst.col_id();

      if (cur_regst_desc_id == in_regst_desc_id_) {
        if (cur_col_id == 0) { in_regsts_.push_back(std::stack<Regst*>()); }
        in_regsts_.back().push(cur_regst);
      } else if (cur_regst_desc_id == out_regst_desc_id_) {
        if (cur_col_id == 0) {
          out_regsts_.push_back(std::deque<Regst*>{cur_regst});
          if (out_regsts_.size() == 1) {
            AsyncReturnModelRegstUntilMatchCurOutRegst();
          }
        } else {
          out_regsts_.back().push_back(cur_regst);
        }
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        init_hid_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == out_diff_regst_desc_id_) {
        if (insert_order_ == 0) {
          if (cur_col_id == 0) {
            insert_order_ = 1;
          } else if (cur_pst.IsLastCol()) {
            insert_order_ = -1;
          } else {
            UNEXPECTED_RUN();
          }
        }

        if ((insert_order_ == 1 && cur_col_id == 0)
            || (insert_order_ == -1 && cur_pst.IsLastCol())) {
          out_diff_regsts_.push_back(std::deque<Regst*>());
        }
        if (insert_order_ == 1) {
          out_diff_regsts_.back().push_back(cur_regst);
        } else {
          out_diff_regsts_.back().push_front(cur_regst);
        }
      } else if (cur_regst_desc_id == rec_acc_diff_regst_desc_id_) {
        CHECK(!rec_acc_diff_regst_);
        if (cur_col_id == 0) {
          AsyncSendRegstMsgToProducer(cur_regst);
        } else {
          rec_acc_diff_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        model_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = cur_regst;
      } else if (cur_regst_desc_id == data_tmp_regst_desc_id_) {
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
      || model_regsts_.empty() || !model_tmp_regst_
      || data_tmp_regsts_.empty()) {
    return false;
  }

  Regst* out_regst = out_regsts_.front().back();
  const PieceStatus& cur_pst = out_regst->piece_status();

  if (cur_pst.col_id() == 0) {
    if (initial_hidden_regst_desc_id_ != -1 && init_hid_regsts_.empty()) {
      return false;
    }
  }

  if (cur_pst.IsLastCol()) {
    CHECK(!rec_acc_diff_regst_);
    if (!(in_regsts_.front().top()->piece_status().IsLastCol())) {
      return false;
    }
    if (!(out_diff_regsts_.front().back()->piece_status().IsLastCol())) {
      return false;
    }
    if (!(data_tmp_regsts_.front().top()->piece_status().IsLastCol())) {
      return false;
    }
  } else {
    if (!rec_acc_diff_regst_) { return false; }
    CHECK(rec_acc_diff_regst_->piece_status().IsNextColOf(cur_pst));
    CHECK(in_regsts_.front().top()->piece_status() == cur_pst);
    CHECK(out_diff_regsts_.front().back()->piece_status() == cur_pst);
    CHECK(data_tmp_regsts_.front().top()->piece_status() == cur_pst);
  }
  return true;
}

void RecurrentBackwardCompActor::AsyncReturnModelRegstUntilMatchCurOutRegst() {
  const Regst* out_regst = out_regsts_.front().front();
  int64_t cur_model_vid = out_regst->model_version_id();
  while (!model_regsts_.empty()
         && model_regsts_.front()->model_version_id() < cur_model_vid) {
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
  if (!model_regsts_.empty()) {
    CHECK_EQ(cur_model_vid, model_regsts_.front()->model_version_id());
  }
}

void RecurrentBackwardCompActor::Act() {
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this](int64_t regst_desc_id) -> Regst* { return nullptr; });
  AsyncSendRegstMsgToConsumer([](Regst* regst) { return true; });

  Regst* out_regst = out_regsts_.front().back();
  out_regsts_.front().pop_back();

  AsyncSendRegstMsgToProducer(in_regsts_.front().top());
  in_regsts_.front().pop();
  AsyncSendRegstMsgToProducer(out_diff_regsts_.front().back());
  out_diff_regsts_.front().pop_back();
  AsyncSendRegstMsgToProducer(data_tmp_regsts_.front().top());
  data_tmp_regsts_.front().pop();

  if (!(out_regst->piece_status().IsLastCol())) {
    AsyncSendRegstMsgToProducer(rec_acc_diff_regst_);
    rec_acc_diff_regst_ = nullptr;
  }
  if (out_regst->piece_status().col_id() == 0) {
    AsyncSendRegstMsgToProducer(init_hid_regsts_.front());
    init_hid_regsts_.pop();

    CHECK(in_regsts_.front().empty());
    in_regsts_.pop_front();
    CHECK(out_diff_regsts_.front().empty());
    out_diff_regsts_.pop_front();
    CHECK(data_tmp_regsts_.front().empty());
    data_tmp_regsts_.pop_front();
    CHECK(out_regsts_.front().empty());
    out_regsts_.pop_front();

    if (out_regsts_.empty()) {
      int64_t last_pid = GetLastPieceIdForModelVersionId(
          model_regsts_.front()->model_version_id());
      if (out_regst->piece_status().piece_id() == last_pid) {
        AsyncSendRegstMsgToProducer(model_regsts_.front());
        model_regsts_.pop();
      }
    } else {
      AsyncReturnModelRegstUntilMatchCurOutRegst();
    }
  }
  AsyncSendRegstMsgToProducer(out_regst);
}

bool RecurrentBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_ && out_diff_regsts_.empty();
}

void RecurrentBackwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(in_regsts_.empty());
  CHECK(out_regsts_.empty());
  CHECK(data_tmp_regsts_.empty());
  CHECK(out_diff_regsts_.empty());
  CHECK(init_hid_regsts_.empty());
  CHECK(!rec_acc_diff_regst_);

  CHECK(model_tmp_regst_);
  AsyncSendRegstMsgToProducer(model_tmp_regst_);

  while (!model_regsts_.empty()) {
    AsyncSendRegstMsgToProducer(model_regsts_.front());
    model_regsts_.pop();
  }
}

REGISTER_ACTOR(TaskType::kRecurrentBackward, RecurrentBackwardCompActor);

}  // namespace oneflow
