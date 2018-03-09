#include "oneflow/core/actor/recurrent_backward_compute_actor.h"

namespace oneflow {

void RecurrentBackwardCompActor::VirtualBackwardCompActorInit(
    const TaskProto& task_proto) {
  h0_regst_desc_id_ = RegstDescId4Name("h0");
  rec_in_regst_desc_id_ = RegstDescId4Name("rec_in");
  rec_out_diff_regst_desc_id_ = RegstDescId4Name("rec_out_diff");

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    if (pair.second == model_regst_desc_id()
        || pair.second == model_tmp_regst_desc_id()
        || pair.second == h0_regst_desc_id_
        || pair.second == rec_out_diff_regst_desc_id_) {
      continue;
    }
    readable_deq_regsts_[pair.second] = {};
  }
  if (parallel_ctx()->policy() == kDataParallel) {
    CHECK_EQ(-1, rec_out_diff_regst_desc_id_);
    CHECK_EQ(-1, rec_in_regst_desc_id_);
  } else {
    CHECK_NE(-1, rec_out_diff_regst_desc_id_);
    CHECK_NE(-1, rec_in_regst_desc_id_);
  }
  rec_out_diff_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&RecurrentBackwardCompActor::HandlerNormal);
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

      if (cur_regst_desc_id == model_regst_desc_id()) {
        model_regsts()->push(cur_regst);
      } else if (cur_regst_desc_id == model_tmp_regst_desc_id()) {
        CHECK(!model_tmp_regst());
        set_model_tmp_regst(cur_regst);
      } else if (cur_regst_desc_id == h0_regst_desc_id_) {
        h0_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == rec_out_diff_regst_desc_id_) {
        CHECK(!rec_out_diff_regst_);
        if (cur_col_id == 0) {
          AsyncSendRegstMsgToProducer(cur_regst);
        } else {
          rec_out_diff_regst_ = cur_regst;
        }
      } else {
        auto& rdq = readable_deq_regsts_.at(cur_regst_desc_id);
        if (cur_regst_desc_id == out_diff_regst_desc_id()) {
          HandleOutDiffRegsts(cur_regst, &rdq);
        } else {
          if (cur_regst_desc_id == rec_in_regst_desc_id_
              && cur_regst->IsMaxCol()) {
            AsyncSendRegstMsgToProducer(cur_regst);
          } else {
            if (cur_col_id == 0) {
              rdq.push_back(std::deque<Regst*>());
              if (cur_regst_desc_id == out_regst_desc_id() && rdq.size() == 1) {
                AsyncReturnModelRegstUntilMatchCurOutRegst(
                    cur_regst->model_version_id());
              }
            }
            rdq.back().push_back(cur_regst);
          }
        }
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool RecurrentBackwardCompActor::RetFalseOrTerminate(Regst* out_regst) const {
  if (out_regst->IsMaxCol()) {
    return false;
  } else {
    UNEXPECTED_RUN();
  }
}

bool RecurrentBackwardCompActor::IsReadReady() {
  if (model_regsts()->empty()) { return false; }
  if (model_tmp_regst_desc_id() != -1 && !model_tmp_regst()) { return false; }

  auto& out_rdq = readable_deq_regsts_.at(out_regst_desc_id());
  if (out_rdq.empty()) { return false; }
  CHECK(!out_rdq.front().empty());
  Regst* out_regst = out_rdq.front().back();
  if (!out_regst->IsMaxCol() && !has_cur_piece_started()) { return false; }

  if (out_regst->col_id() == 0 && h0_regst_desc_id_ != -1
      && h0_regsts_.empty()) {
    return false;
  }

  bool is_model_parallel = parallel_ctx()->policy() == kModelParallel;
  if (is_model_parallel && !out_regst->IsMaxCol()) {
    if (!rec_out_diff_regst_) { return false; }
    CHECK(rec_out_diff_regst_->HaveNextPieceColStatusOf(out_regst));
  }

  for (auto& pair : readable_deq_regsts_) {
    if (pair.first == out_regst_desc_id()) { continue; }
    auto& rdq = pair.second;
    if (rdq.empty()) { return false; }
    if (pair.first == rec_in_regst_desc_id_) {
      if (out_regst->col_id() != 0
          && !out_regst->HaveNextPieceColStatusOf(rdq.front().back())) {
        return RetFalseOrTerminate(out_regst);
      }
    } else {
      if (rdq.front().empty()) { return false; }  // for out_diff
      if (!out_regst->HaveSamePieceColStatusAs(out_regst)) {
        return RetFalseOrTerminate(out_regst);
      }
    }
  }
  return true;
}

Blob* RecurrentBackwardCompActor::HandleSpecialBnInOp(
    const std::string& bn_in_op) {
  if (bn_in_op == "rec_in" && parallel_ctx()->policy() == kDataParallel) {
    auto& out_rdq = readable_deq_regsts_.at(out_regst_desc_id());
    CHECK_GT(out_rdq.front().back()->col_id(), 0);
    Regst* regst = *(out_rdq.front().end() - 2);
    return regst->GetBlobByLbn("rnn_cell/out");
  }
  return nullptr;
}

void RecurrentBackwardCompActor::Act() {
  AsyncLaunchKernel(
      GenDefaultKernelCtx(), [this](int64_t regst_desc_id) -> Regst* {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          if (regst_desc_id == model_regst_desc_id()) {
            return model_regsts()->front();
          } else if (regst_desc_id == model_tmp_regst_desc_id()) {
            return model_tmp_regst();
          } else if (regst_desc_id == h0_regst_desc_id_) {
            return h0_regsts_.front();
          } else if (regst_desc_id == rec_out_diff_regst_desc_id_) {
            return rec_out_diff_regst_;
          } else {
            return readable_deq_regsts_.at(regst_desc_id).front().back();
          }
        } else {
          return regst;
        }
      });

  auto& out_rdq = readable_deq_regsts_.at(out_regst_desc_id());
  Regst* out_regst = out_rdq.front().back();
  out_rdq.front().pop_back();
  if (out_regst->col_id() == 0) {
    CHECK(out_rdq.front().empty());
    out_rdq.pop_front();
  }

  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(out_regst->piece_id());
    regst->set_col_id(out_regst->col_id());
    regst->set_max_col_id(out_regst->max_col_id());
    return true;
  });
  // model
  if (out_rdq.empty()) {
    AsyncReturnModelRegstUntilLastPieceIdGreaterThan(out_regst->piece_id());
  } else {
    CHECK(!out_rdq.front().empty());
    AsyncReturnModelRegstUntilMatchCurOutRegst(
        out_rdq.front().back()->model_version_id());
  }
  // h0
  if (out_regst->col_id() == 0 && h0_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(h0_regsts_.front());
    h0_regsts_.pop();
  }
  // rec_out_diff
  if (!out_regst->IsMaxCol() && rec_out_diff_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(rec_out_diff_regst_);
    rec_out_diff_regst_ = nullptr;
  }
  // other
  for (auto& pair : readable_deq_regsts_) {
    if (pair.first == out_regst_desc_id()) { continue; }
    auto& rdq = pair.second;
    if ((pair.first != rec_in_regst_desc_id_)
        || (pair.first == rec_in_regst_desc_id_ && out_regst->col_id() > 0)) {
      AsyncSendRegstMsgToProducer(rdq.front().back());
      rdq.front().pop_back();
    }
    if (out_regst->col_id() == 0) {
      CHECK(rdq.front().empty());
      rdq.pop_front();
    }
  }

  if (out_regst->IsMaxCol()) { set_has_cur_piece_started(true); }
  if (out_regst->col_id() == 0) { set_has_cur_piece_started(false); }
  AsyncSendRegstMsgToProducer(out_regst);
}

bool RecurrentBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord()
         && readable_deq_regsts_.at(out_diff_regst_desc_id()).empty();
}

void RecurrentBackwardCompActor::CheckBeforeAsyncReturnAllReadableRegst() {
  CHECK(h0_regsts_.empty());
  CHECK(!rec_out_diff_regst_);
  for (auto& pair : readable_deq_regsts_) { CHECK(pair.second.empty()); }
}

void RecurrentBackwardCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  Regst* out_regst =
      readable_deq_regsts_.at(out_regst_desc_id()).front().back();
  handler(out_regst);
  if (!out_regst->IsMaxCol() && rec_out_diff_regst_desc_id_ != -1) {
    handler(rec_out_diff_regst_);
  }
  if (out_regst->col_id() == 0 && h0_regst_desc_id_ != -1) {
    handler(h0_regsts_.front());
  }
  for (const auto& pair : readable_deq_regsts_) {
    if (pair.first == out_regst_desc_id()) { continue; }
    if ((pair.first != rec_in_regst_desc_id_)
        || (pair.first == rec_in_regst_desc_id_ && out_regst->col_id() != 0)) {
      handler(pair.second.front().back());
    }
  }
  ForCurReadableModelAndModelTmp(handler);
}

REGISTER_ACTOR(TaskType::kRecurrentBackward, RecurrentBackwardCompActor);

}  // namespace oneflow
