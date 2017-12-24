#include "oneflow/core/actor/basic_rnn_backward_compute_actor.h"

namespace oneflow {

void BasicRnnBackwardCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  in_regst_desc_id_ = RegstDescId4Name("in");
  out_regst_desc_id_ = RegstDescId4Name("out");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  rec_acc_diff_regst_desc_id_ = RegstDescId4Name("rec_acc_diff");
  model_regst_desc_id_ = RegstDescId4Name("model");
  activation_regst_desc_id_ = RegstDescId4Name("activation");

  is_out_diff_eord_ = false;
  is_insert_to_back_ = true;
  DecreaseRemainingEordCnt();  // no 'rec_acc_diff', else will cause deadlock
  OF_SET_MSG_HANDLER(&BasicRnnBackwardCompActor::HandlerNormal);
}

int BasicRnnBackwardCompActor::HandlerNormal(const ActorMsg& msg) {
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
      int64_t cur_pid = cur_pst.piece_id();
      int64_t cur_col_id = cur_pst.col_id();
      int64_t cur_model_vid = cur_regst->model_version_id();

      if (cur_regst_desc_id == in_regst_desc_id_) {
        pid2in_regsts_[cur_pid].push(cur_regst);  // insert or push
      } else if (cur_regst_desc_id == out_regst_desc_id_) {
        pid2out_regsts_[cur_pid].push_back(cur_regst);  // insert or pushback
        if (cur_col_id == 0) {
          model_vid2cnt_[cur_model_vid] += 1;        // insert or add
          model_vid2status_[cur_model_vid] = false;  // insert or set

          if ((cur_model_vid > 0)
              && (model_vid2status_.find(cur_model_vid - 1)
                  != model_vid2status_.end())) {
            model_vid2status_.at(cur_model_vid - 1) = true;
            if (model_vid2cnt_.find(cur_model_vid - 1)
                == model_vid2cnt_.end()) {
              RelModelByJudgingStatus(cur_model_vid - 1);
            }
          }

          if (cur_pid == GetLastPieceIdForModelVersionId(cur_model_vid)) {
            model_vid2status_.at(cur_model_vid) = true;
          }
          if (cur_pid == RuntimeCtx::Singleton()->total_piece_num() - 1) {
            model_vid2status_.at(cur_model_vid) = true;
          }
        }
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        CHECK(pid2init_hid_regsts_.emplace(cur_pid, cur_regst).second);
      } else if (cur_regst_desc_id == out_diff_regst_desc_id_) {
        auto it = pid2out_diff_regsts_.find(cur_pid);
        if (it == pid2out_diff_regsts_.end()) {
          if (cur_col_id == 0) {
            is_insert_to_back_ = true;
          } else if (cur_pst.IsLastCol()) {
            is_insert_to_back_ = false;
          } else {
            // do nothing
          }
        }
        if (is_insert_to_back_) {
          pid2out_diff_regsts_[cur_pid].push_back(cur_regst);  // insert or push
        } else {
          pid2out_diff_regsts_[cur_pid].push_front(cur_regst);
        }
      } else if (cur_regst_desc_id == rec_acc_diff_regst_desc_id_) {
        CHECK_EQ(-1, cur_regst->recurrent_flag());
        CHECK(pid2rec_acc_diff_regsts_.emplace(cur_pid, cur_regst).second);
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        CHECK(model_vid2model_regst_.emplace(cur_model_vid, cur_regst).second);
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool BasicRnnBackwardCompActor::CheckModel_In_OutDiff_Activation(
    Regst* out_regst) const {
  const PieceStatus& cur_pst = out_regst->piece_status();
  int64_t cur_pid = cur_pst.piece_id();
  int64_t cur_model_vid = out_regst->model_version_id();

  auto model_it = model_vid2model_regst_.find(cur_model_vid);
  if (model_it == model_vid2model_regst_.end()) { return false; }

  auto in_it = pid2in_regsts_.find(cur_pid);
  if (in_it == pid2in_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (in_it->second.top()->piece_status() != cur_pst) { return false; }
  } else {
    CHECK(in_it->second.top()->piece_status() == cur_pst);
  }

  auto out_diff_it = pid2out_diff_regsts_.find(cur_pid);
  if (out_diff_it == pid2out_diff_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (out_diff_it->second.back()->piece_status() != cur_pst) { return false; }
  } else {
    CHECK(out_diff_it->second.back()->piece_status() == cur_pst);
  }

  auto act_it = pid2activation_regsts_.find(cur_pid);
  if (act_it == pid2activation_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (act_it->second.top()->piece_status() != cur_pst) { return false; }
  } else {
    CHECK(act_it->second.top()->piece_status() == cur_pst);
  }

  return true;
}

void BasicRnnBackwardCompActor::FillReadableWithIn_OutDiff_Model_Activation(
    Regst* out_regst) {
  int64_t cur_pid = out_regst->piece_status().piece_id();
  int64_t cur_model_vid = out_regst->model_version_id();
  readable_regsts_.emplace(in_regst_desc_id_, pid2in_regsts_.at(cur_pid).top());
  readable_regsts_.emplace(out_diff_regst_desc_id_,
                           pid2out_diff_regsts_.at(cur_pid).back());
  readable_regsts_.emplace(model_regst_desc_id_,
                           model_vid2model_regst_.at(cur_model_vid));
  readable_regsts_.emplace(activation_regst_desc_id_,
                           pid2activation_regsts_.at(cur_pid).top());
}

bool BasicRnnBackwardCompActor::IsReadReady() {
  if (pid2in_regsts_.empty() || pid2out_regsts_.empty()
      || pid2out_diff_regsts_.empty() || model_vid2model_regst_.empty()
      || pid2activation_regsts_.empty()) {
    return false;
  }
  for (const auto& kv : pid2out_regsts_) {
    Regst* out_regst = kv.second.back();
    const PieceStatus& cur_pst = out_regst->piece_status();
    int64_t cur_pid = cur_pst.piece_id();

    if (!CheckModel_In_OutDiff_Activation(out_regst)) { continue; }

    readable_regsts_.clear();
    if (cur_pst.col_id() == 0) {
      auto init_hid_it = pid2init_hid_regsts_.find(cur_pid);
      if (init_hid_it == pid2init_hid_regsts_.end()) { continue; }
      readable_regsts_.emplace(initial_hidden_regst_desc_id_,
                               init_hid_it->second);
    } else {
      readable_regsts_.emplace(out_regst_desc_id_,
                               *(pid2out_regsts_.at(cur_pid).end() - 2));
    }
    if (!cur_pst.IsLastCol()) {
      auto rec_acc_it = pid2rec_acc_diff_regsts_.find(cur_pid);
      if (rec_acc_it == pid2rec_acc_diff_regsts_.end()) { continue; }
      CHECK(rec_acc_it->second->piece_status().IsNextColOf(
          out_regst->piece_status()));
      readable_regsts_.emplace(rec_acc_diff_regst_desc_id_, rec_acc_it->second);
    } else {
      CHECK_EQ(kv.second.size(), pid2out_regsts_.at(cur_pid).size());
      CHECK_EQ(kv.second.size(), pid2activation_regsts_.at(cur_pid).size());
    }
    FillReadableWithIn_OutDiff_Model_Activation(out_regst);
    return true;
  }
  return false;
}

bool BasicRnnBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_ && pid2out_diff_regsts_.empty();
}

void BasicRnnBackwardCompActor::RelModelByJudgingStatus(int64_t model_vid) {
  if (model_vid2status_.at(model_vid)) {
    AsyncSendRegstMsgToProducer(model_vid2model_regst_.at(model_vid));
    model_vid2model_regst_.erase(model_vid);
    model_vid2status_.erase(model_vid);
  }
}

void BasicRnnBackwardCompActor::UpdtModelStatusAfterAct() {
  Regst* out_diff_regst = readable_regsts_.at(out_diff_regst_desc_id_);
  const PieceStatus& cur_pst = out_diff_regst->piece_status();
  int64_t cur_col_id = cur_pst.col_id();
  Regst* model_regst = readable_regsts_.at(model_regst_desc_id_);
  int64_t cur_model_vid = model_regst->model_version_id();

  if (cur_col_id == 0) {
    model_vid2cnt_.at(cur_model_vid) -= 1;
    if (model_vid2cnt_.at(cur_model_vid) == 0) {
      model_vid2cnt_.erase(cur_model_vid);
      RelModelByJudgingStatus(cur_model_vid);
    }
  }
}

void BasicRnnBackwardCompActor::Act() {
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this](int64_t regst_desc_id) -> Regst* { return nullptr; });
  AsyncSendRegstMsgToConsumer([](Regst* regst) {
    regst->set_is_forward(false);
    return true;
  });

  Regst* out_diff_regst = readable_regsts_.at(out_diff_regst_desc_id_);
  const PieceStatus& cur_pst = out_diff_regst->piece_status();
  int64_t cur_pid = cur_pst.piece_id();
  int64_t cur_col_id = cur_pst.col_id();
  Regst* model_regst = readable_regsts_.at(model_regst_desc_id_);

  UpdtModelStatusAfterAct();

#define ERASE_ELES_IN_HASHMAP_WHEN_COL0(hash_map) \
  do {                                            \
    if (cur_col_id == 0) {                        \
      CHECK(hash_map.at(cur_pid).empty());        \
      hash_map.erase(cur_pid);                    \
    }                                             \
  } while (0)

  // update out_regst
  // the out_regst inserted to readable_regsts_ is not back(), but 'back()-1'
  CHECK(pid2out_regsts_.at(cur_pid).back()->piece_status() == cur_pst);
  AsyncSendRegstMsgToProducer(pid2out_regsts_.at(cur_pid).back());
  pid2out_regsts_.at(cur_pid).pop_back();
  ERASE_ELES_IN_HASHMAP_WHEN_COL0(pid2out_regsts_);

  for (auto& kv : readable_regsts_) {
    if (kv.first == model_regst_desc_id_) { continue; }
    if (kv.first == out_regst_desc_id_) { continue; }
    AsyncSendRegstMsgToProducer(kv.second);

    if (kv.first == in_regst_desc_id_) {
      pid2in_regsts_.at(cur_pid).pop();
      ERASE_ELES_IN_HASHMAP_WHEN_COL0(pid2in_regsts_);
    } else if (kv.first == out_diff_regst_desc_id_) {
      pid2out_diff_regsts_.at(cur_pid).pop_back();
      if (pid2out_diff_regsts_.at(cur_pid).empty()) {
        pid2out_diff_regsts_.erase(cur_pid);
      }
    } else if (kv.first == initial_hidden_regst_desc_id_) {
      CHECK_EQ(0, cur_col_id);
      pid2init_hid_regsts_.erase(cur_pid);
    } else if (kv.first == rec_acc_diff_regst_desc_id_) {
      CHECK(!cur_pst.IsLastCol());
      pid2rec_acc_diff_regsts_.erase(cur_pid);
    } else if (kv.first == activation_regst_desc_id_) {
      pid2activation_regsts_.at(cur_pid).pop();
      ERASE_ELES_IN_HASHMAP_WHEN_COL0(pid2activation_regsts_);
#undef ERASE_ELES_IN_HASHMAP_WHEN_COL0
    } else {
      UNEXPECTED_RUN();
    }
  }
}

void BasicRnnBackwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(pid2in_regsts_.empty());
  CHECK(pid2out_regsts_.empty());
  CHECK(pid2out_diff_regsts_.empty());
  CHECK(pid2init_hid_regsts_.empty());
  CHECK(pid2rec_acc_diff_regsts_.empty());
  CHECK(pid2activation_regsts_.empty());
  CHECK(model_vid2cnt_.empty());
  CHECK(model_vid2status_.empty());
  for (auto& kv : model_vid2model_regst_) {
    AsyncSendRegstMsgToProducer(kv.second);
  }
  model_vid2model_regst_.clear();
}

REGISTER_ACTOR(TaskType::kBasicRnnBackward, BasicRnnBackwardCompActor);

}  // namespace oneflow
