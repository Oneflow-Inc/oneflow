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
  // model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  // activation_regst_desc_id_ = RegstDescId4Name("activation");
  // data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");

  expected_initial_hidden_piece_id_ = 0;
  expected_model_version_id_ = 0;
  model_tmp_regst_ = nullptr;
  is_insert_to_back_ = true;

  // not consider rec_acc_diff, else will cause deadlock
  set_num_of_remaining_eord(3 + (initial_hidden_regst_desc_id_ != -1)
                            + (model_regst_desc_id_ != -1));
                            // + (model_tmp_regst_desc_id_ != -1)
                            // + (activation_regst_desc_id_ != -1)
                            // + (data_tmp_regst_desc_id_ != -1));
  mut_num_of_read_empty() = 1;  // only consider in
  
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  OF_SET_MSG_HANDLER(&BpDataCompActor::HandlerNormal);
}

bool RnnBpDataCompActor::CheckModel_Out_OutDiff(Regst* cur_regst) const {
  const PieceStatus& cur_pst = cur_regst->piece_status();
  int64_t cur_pid = cur_pst.piece_id();
  int64_t cur_col_id = cur_pst.col_id();
  int64_t cur_model_vid = cur_regst->model_version_id();

  auto model_it = model_vid2model_regst_.find(cur_model_vid);
  if (model_it == model_vid2model_regst_.end()) { return false; }

  auto out_it = pid2out_regsts_.find(cur_pid);
  if (out_it == pid2out_regsts_.end()) { return false; }
  if (cur_pst.IsLastCol()) {
    if (!out_it->second.top()->piece_status().IsLastCol()) { return false; }
  } else {
    CHECK_EQ(cur_col_id, out_it->second.top()->piece_status().col_id());
  }

  auto out_diff_it = pid2out_diff_regsts_.find(cur_pid);
  if (out_diff_it == pid2out_diff_regsts_.end()) { return false; }
  if (out_diff_it->second.back()->piece_status().col_id() != cur_col_id) { return false; }

  return true;
}

void RnnBpDataCompActor::FillMatl4ActWithIn_Out_OutDiff_Model(Regst* cur_regst) {
  int64_t cur_pid = cur_regst->piece_status().piece_id();
  int64_t cur_model_vid = cur_regst->model_version_id();

  matl4act_.readable_regst_.clear();  //TODO: clear here or after act
  matl4act_.readable_regst_.emplace(in_regst_desc_id_, cur_regst);
  matl4act_.readable_regst_.empalce(out_regst_desc_id_, 
                                    pid2out_regsts_.at(cur_pid).back());
  matl4act_.readable_regst_.empalce(out_diff_regst_desc_id_, 
                                    pid2out_diff_regsts_.at(cur_pid).back());
  matl4act_.readable_regst_.empalce(model_regst_desc_id_, 
                                    model_vid2model_regst_.at(cur_model_vid));
}

bool RnnBpDataCompActor::IsReadReady() {
  if (pid2in_regsts_.empty() 
      || pid2out_regsts_.empty() 
      || pid2out_diff_regsts_.empty()
      || model_vid2model_regst_.empty()) {
    return false;
  }

  // RnnCell
  
  for (auto& kv : pid2in_regsts_) {
    Regst* cur_regst = kv.second.top();
    PieceStatus& cur_pst = cur_regst->piece_status();
    int64_t cur_pid = cur_pst.piece_id();
    int64_t cur_model_vid = cur_regst->model_version_id();

    if (!CheckModel_Out_OutDiff(cur_regst)) { continue; }
    if (cur_pst.col_id() == 0) {
      auto init_hid_it = pid2init_hid_regsts_.find(cur_pid);
      if (init_hid_it == pid2init_hid_regsts_.end()) { continue; }
      matl4act_.readable_regst_.empalce(initial_hidden_regst_desc_id_, init_hid_it->second);
    } else {
      const auto out_it = pid2out_regsts_.find(cur_pid);
      const auto list_it = out_it->second.cend();
      list_it --;
      matl4act_.pre_out_regst = *list_it;
    }
    if (!cur_pst.IsLastCol()) {
      auto rec_acc_it = pid2rec_acc_diff_regsts_.find(cur_pid);
      if (rec_acc_it == pid2rec_acc_diff_regsts_.end()) { continue; }
      CHECK(rec_acc_it->second->piece_status().IsNextColOf(cur_regst->piece_status()));
      matl4act_.readable_regst_.emplace(rec_acc_diff_regst_desc_id_, rec_acc_it->second);
    } else {
      CHECK_EQ(kv.second.size(), pid2out_regsts_.at(cur_pid).size());
    }
    FillMatl4ActWithIn_Out_OutDiff_Model(cur_regst);
    return true;
  }
  return false;
}

void RnnBpDataCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
  if (model_regst_desc_id_ != -1) {
    for (auto& kv : model_vid2model_regst_) {
      AsyncSendRegstMsgToProducer(kv.second);
    }
    model_vid2model_regst_.clear();
  }
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
  }
}

int RnnBpDataCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(mgs.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
    if (msg_handler() == &RnnBpDataCompActor::HandlerZombie
        || msg_handler() == nullptr) {
      CHECK_EQ(pid2in_regsts_.empty());
      CHECK_EQ(pid2out_regsts_.empty());
      CHECK_EQ(pid2init_hid_regsts_.empty());
      CHECK_EQ(pid2out_diff_regsts_.empty());
      CHECK_EQ(pid2rec_acc_diff_regsts_.empty());
      CHECK_EQ(model_vid2status_.empty());
      AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t cur_pid = regst->piece_status().piece_id();
      int64_t regst_desc_id = regst->regst_desc_id();
      int64_t model_vid = regst->model_version_id();
      int64_t col_id = regst->piece_status().col_id();

      if (regst_desc_id == in_regst_desc_id_) {
        CHECK(regst->piece_status() == expected_piece_status_);
        if (pid2in_regsts_.empty()) { mut_num_of_read_empty() = 0; }
        pid2in_regsts_[cur_pid].push(regst);  // insert or push
        expected_piece_status_.GetIntoNextStatus();

        if (col_id == 0) {
          int64_t model_vid_to_be_set = -1;
          // mark pre-model as no-more-new-piece  
          if (model_vid > 0) {
            model_vid_to_be_set = model_vid - 1;
          }
          // mark model of last-pid as no-more-new-piece
          if (cur_pid == JobDesc::Singleton()->total_piece_num() - 1) {
            model_vid_to_be_set = model_vid;
          }

          if (model_vid_to_be_set != -1) {
            model_vid2status_[model_vid_to_be_set] = true;  // insert or set
          }
        }
      } else if (regst_desc_id == out_regst_desc_id_) {
        pid2out_regsts_[cur_pid].push_back(regst);  // insert or push_back
      } else if (regst_desc_id == initial_hidden_regst_desc_id_) {
        CHECK_EQ(expected_initial_hidden_piece_id_, cur_pid);
        CHECK(pid2init_hid_regsts_.emplace(cur_pid, regst).second);
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
          pid2out_diff_regsts_[cur_pid].push_front(regst); // insert or push
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
      // } else if (regst_desc_id == model_tmp_regst_desc_id_) {
      //   CHECK(!model_tmp_regst_);
      //   model_tmp_regst_ = regst;
      // } else if (regst_desc_id == activation_regst_desc_id_) {
      //   activation_regsts_.push(regst);
      // } else if (regst_desc_id == data_tmp_regst_desc_id_) {
      //   data_tmp_regsts_.push(regst);
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

} // namespace oneflow
