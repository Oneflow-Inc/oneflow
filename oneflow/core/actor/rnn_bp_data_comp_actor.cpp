#include "oneflow/core/actor/rnn_bp_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void RnnBpDataCompActor::Init(const TaskProto& task_proto,
                              const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);

  in_regst_desc_id_ = RegstDescId4Name("in");
  out_regst_desc_id_ = RegstDescId4Name("out");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  rec_acc_diff_regst_desc_id_ = RegstDescId4Name("rec_acc_diff");
  model_regst_desc_id_ = RegstDescId4Name("model");

  expected_initial_hidden_piece_id_ = 0;
  expected_model_version_id_ = 0;
  is_insert_to_back_ = true;

  // not consider rec_acc_diff, else will cause deadlock
  set_num_of_remaining_eord(5);
  mut_num_of_read_empty() = 1;  // only consider in

  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  OF_SET_MSG_HANDLER(&RnnBpDataCompActor::HandlerNormal);
}

bool RnnBpDataCompActor::CheckModel_Out_OutDiff(Regst* cur_regst) const {
  const PieceStatus& cur_pst = cur_regst->piece_status();
  int64_t cur_pid = cur_pst.piece_id();
  int64_t cur_model_vid = cur_regst->model_version_id();

  auto model_it = model_vid2model_regst_.find(cur_model_vid);
  if (model_it == model_vid2model_regst_.end()) { return false; }

  auto out_it = pid2out_regsts_.find(cur_pid);
  if (out_it == pid2out_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (out_it->second.back()->piece_status() != cur_pst) { return false; }
  } else {
    CHECK(out_it->second.back()->piece_status() == cur_pst);
  }

  auto out_diff_it = pid2out_diff_regsts_.find(cur_pid);
  if (out_diff_it == pid2out_diff_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (out_diff_it->second.back()->piece_status() != cur_pst) { return false; }
  } else {
    CHECK(out_diff_it->second.back()->piece_status() == cur_pst);
  }

  return true;
}

void RnnBpDataCompActor::FillMatl4ActWithIn_Out_OutDiff_Model(
    Regst* cur_regst) {
  int64_t cur_pid = cur_regst->piece_status().piece_id();
  int64_t cur_model_vid = cur_regst->model_version_id();

  matl4act_.readable_regsts_.emplace(in_regst_desc_id_, cur_regst);
  matl4act_.readable_regsts_.emplace(out_regst_desc_id_,
                                     pid2out_regsts_.at(cur_pid).back());
  matl4act_.readable_regsts_.emplace(out_diff_regst_desc_id_,
                                     pid2out_diff_regsts_.at(cur_pid).back());
  matl4act_.readable_regsts_.emplace(model_regst_desc_id_,
                                     model_vid2model_regst_.at(cur_model_vid));
}

bool RnnBpDataCompActor::IsReadReady() {
  if (pid2in_regsts_.empty() || pid2out_regsts_.empty()
      || pid2out_diff_regsts_.empty() || model_vid2model_regst_.empty()) {
    return false;
  }

  for (auto& kv : pid2in_regsts_) {
    Regst* cur_regst = kv.second.top();
    const PieceStatus& cur_pst = cur_regst->piece_status();
    int64_t cur_pid = cur_pst.piece_id();

    if (!CheckModel_Out_OutDiff(cur_regst)) { continue; }

    matl4act_.readable_regsts_.clear();
    if (cur_pst.col_id() == 0) {
      auto init_hid_it = pid2init_hid_regsts_.find(cur_pid);
      if (init_hid_it == pid2init_hid_regsts_.end()) { continue; }
      matl4act_.readable_regsts_.emplace(initial_hidden_regst_desc_id_,
                                         init_hid_it->second);
    } else {
      auto out_it = pid2out_regsts_.find(cur_pid);
      auto list_it = out_it->second.cend();
      --list_it;
      --list_it;
      matl4act_.pre_out_regst = *list_it;
    }
    if (!cur_pst.IsLastCol()) {
      auto rec_acc_it = pid2rec_acc_diff_regsts_.find(cur_pid);
      if (rec_acc_it == pid2rec_acc_diff_regsts_.end()) { continue; }
      CHECK(rec_acc_it->second->piece_status().IsNextColOf(
          cur_regst->piece_status()));
      matl4act_.readable_regsts_.emplace(rec_acc_diff_regst_desc_id_,
                                         rec_acc_it->second);
    } else {
      CHECK_EQ(kv.second.size(), pid2out_regsts_.at(cur_pid).size());
    }
    FillMatl4ActWithIn_Out_OutDiff_Model(cur_regst);
    return true;
  }
  return false;
}

void RnnBpDataCompActor::AsyncSendMsgToModelProducer() {
  for (auto& kv : model_vid2model_regst_) {
    AsyncSendRegstMsgToProducer(kv.second);
  }
  model_vid2model_regst_.clear();
  model_vid2status_.clear();
}

int RnnBpDataCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
    if (msg_handler() == &RnnBpDataCompActor::HandlerZombie
        || msg_handler() == nullptr) {
      CHECK(pid2in_regsts_.empty());
      CHECK(pid2out_regsts_.empty());
      CHECK(pid2init_hid_regsts_.empty());
      CHECK(pid2out_diff_regsts_.empty());
      CHECK(pid2rec_acc_diff_regsts_.empty());
      CHECK(model_vid2cnt_.empty());
      // there might be model not been used by any piece, clear it here
      AsyncSendMsgToModelProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t cur_pid = regst->piece_status().piece_id();
      int64_t regst_desc_id = regst->regst_desc_id();
      int64_t model_vid = regst->model_version_id();
      int64_t col_id = regst->piece_status().col_id();

      if (regst_desc_id == in_regst_desc_id_) {
        if (pid2in_regsts_.empty()) { mut_num_of_read_empty() = 0; }
        pid2in_regsts_[cur_pid].push(regst);  // insert or push

        if (col_id == 0) {
          int64_t model_vid_to_be_set = -1;
          // mark pre-model as no-more-new-piece
          if (model_vid > 0) { model_vid_to_be_set = model_vid - 1; }
          // mark model of last-pid as no-more-new-piece
          if (cur_pid == JobDesc::Singleton()->total_piece_num() - 1) {
            model_vid_to_be_set = model_vid;
          }

          if (model_vid_to_be_set != -1) {
            model_vid2status_[model_vid_to_be_set] = true;  // insert or set
          }
        }

      } else if (regst_desc_id == out_regst_desc_id_) {
        if (col_id == 0) {
          model_vid2cnt_[model_vid] += 1;  // insert or add
        }
        pid2out_regsts_[cur_pid].push_back(regst);  // insert or push_back
      } else if (regst_desc_id == initial_hidden_regst_desc_id_) {
        CHECK_EQ(expected_initial_hidden_piece_id_, cur_pid);
        CHECK(pid2init_hid_regsts_.emplace(cur_pid, regst).second);
        expected_initial_hidden_piece_id_ += 1;
      } else if (regst_desc_id == out_diff_regst_desc_id_) {
        auto it = pid2out_diff_regsts_.find(cur_pid);
        if (it == pid2out_diff_regsts_.end()) {
          if (col_id == 0) {
            is_insert_to_back_ = true;
          } else if (regst->piece_status().IsLastCol()) {
            is_insert_to_back_ = false;
          } else {
            // do nothing
          }
        }
        if (is_insert_to_back_) {
          pid2out_diff_regsts_[cur_pid].push_back(regst);  // insert or push
        } else {
          pid2out_diff_regsts_[cur_pid].push_front(regst);  // insert or push
        }
      } else if (regst_desc_id == rec_acc_diff_regst_desc_id_) {
        CHECK_EQ(-1, regst->recurrent_flag());
        CHECK(pid2rec_acc_diff_regsts_.emplace(cur_pid, regst).second);
      } else if (regst_desc_id == model_regst_desc_id_) {
        CHECK_EQ(expected_model_version_id_, model_vid);
        CHECK(model_vid2model_regst_.emplace(model_vid, regst).second);
        if (model_vid2status_.find(model_vid) == model_vid2status_.end()) {
          model_vid2status_.emplace(model_vid, false);
        }
        expected_model_version_id_ += 1;
      } else {
        UNEXPECTED_RUN();
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int RnnBpDataCompActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  if (TryUpdtStateAsProducedRegst(msg.regst()) == 1) {
    CHECK_EQ(-1, msg.regst()->recurrent_flag());
  }
  ActUntilFail();
  if (num_of_read_empty()) {
    AsyncSendMsgToModelProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&RnnBpDataCompActor::HandlerZombie);
  }
  return 0;
}

void RnnBpDataCompActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* { TODO(); });

  const PieceStatus& pst =
      matl4act_.readable_regsts_.at(in_regst_desc_id_)->piece_status();
  AsyncSendRegstMsgToConsumer([pst](Regst* regst) {
    regst->set_piece_status(pst);
    regst->set_is_forward(false);
  });

  // update model_regst
  if (matl4act_.readable_regsts_.at(in_regst_desc_id_)->piece_status().col_id()
      == 0) {
    int64_t cur_model_vid =
        matl4act_.readable_regsts_.at(model_regst_desc_id_)->model_version_id();
    model_vid2cnt_.at(cur_model_vid) -= 1;
    if (model_vid2cnt_.at(cur_model_vid) == 0) {
      model_vid2cnt_.erase(cur_model_vid);
      if (model_vid2status_.at(cur_model_vid)) {
        AsyncSendRegstMsgToProducer(model_vid2model_regst_.at(cur_model_vid));
        model_vid2model_regst_.erase(cur_model_vid);
        model_vid2status_.erase(cur_model_vid);
      }
    }
  }
  // update other regsts
  for (auto& kv : matl4act_.readable_regsts_) {
    const PieceStatus& cur_pst = kv.second->piece_status();
    int64_t cur_pid = cur_pst.piece_id();
    int64_t cur_col_id = cur_pst.col_id();
    if (kv.first != model_regst_desc_id_) {
      AsyncSendRegstMsgToProducer(kv.second);

      if (kv.first == in_regst_desc_id_) {
        pid2in_regsts_.at(cur_pid).pop();
        if (cur_col_id == 0) {
          CHECK(pid2in_regsts_.at(cur_pid).empty());
          pid2in_regsts_.erase(cur_pid);
        }
        mut_num_of_read_empty() = pid2in_regsts_.empty();
      } else if (kv.first == out_regst_desc_id_) {
        pid2out_regsts_.at(cur_pid).pop_back();
        if (cur_col_id == 0) {
          CHECK(pid2out_regsts_.at(cur_pid).empty());
          pid2out_regsts_.erase(cur_pid);
        }
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
      } else {
        UNEXPECTED_RUN();
      }
    }
  }
}

REGISTER_ACTOR(kRnnDataCompTask, false, RnnBpDataCompActor);

}  // namespace oneflow
