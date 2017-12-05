#include "oneflow/core/actor/rnn_fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void RnnFwDataCompActor::Init(const TaskProto& task_proto,
                              const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);

  in_regst_desc_id_ = RegstDescId4Name("in");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  model_regst_desc_id_ = RegstDescId4Name("model");
  out_regst_desc_id_ = RegstDescId4Name("out");

  expected_model_version_id_ = 0;
  expected_initial_hidden_piece_id_ = 0;
  latest_model_regst_ = nullptr;

  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }

  kernel_ctx_ = GenDefaultKernelCtx();
  std::pair<DataLoadBuf, int64_t> ctx =
      std::make_pair(data_load_buf_, parallel_id());
  kernel_ctx_.other = static_cast<void*>(&ctx);

  if (in_regst_desc_id_ == -1) {
    CHECK_EQ(-1, initial_hidden_regst_desc_id_);
    CHECK_EQ(-1, model_regst_desc_id_);
    OF_SET_MSG_HANDLER(&RnnFwDataCompActor::WaitToStart);
  } else {
    // not consider out_regst, else will cause deadlock
    set_num_of_remaining_eord(3);
    mut_num_of_read_empty() = 1;  // only consider in_regst
    OF_SET_MSG_HANDLER(&RnnFwDataCompActor::HandlerNormal);
  }
}

bool RnnFwDataCompActor::ModelSatisfySSP(Regst* in_regst,
                                         Regst* model_regst) const {
  if (JobDesc::Singleton()->is_train()) {
    // Ho Q, Cipar J, Cui H, et al. More effective distributed ml via a stale
    // synchronous parallel parameter server
    int32_t staleness = JobDesc::Singleton()->staleness();
    int32_t num_of_pieces_in_batch =
        JobDesc::Singleton()->num_of_pieces_in_batch();
    int64_t cur_iteration =
        in_regst->piece_status().piece_id() / num_of_pieces_in_batch;
    int64_t stale_version = cur_iteration - staleness;
    return model_regst->model_version_id() >= stale_version;
  } else {
    return true;
  }
}

bool RnnFwDataCompActor::IsReadReady() {
  // DataLoader
  if (in_regst_desc_id_ == -1) { return true; }
  if (in_regst_desc_id_ == -2) {
    return false;
  }  // dataloader but read to the end

  if (pid2in_regsts_.empty()) { return false; }

  // RnnCell
  for (const auto& kv : pid2in_regsts_) {  // increasing order by piece_id
    Regst* cur_regst = kv.second.front();
    const PieceStatus& cur_pst = cur_regst->piece_status();
    int64_t cur_pid = cur_pst.piece_id();
    int64_t cur_col_id = cur_pst.col_id();
    if (cur_col_id == 0) {
      if (initial_hidden_regsts_.empty()) {
        return false;
      } else {  // if not empty, the front of queue must the same pid
        CHECK_EQ(cur_pid,
                 initial_hidden_regsts_.front()->piece_status().piece_id());
        if (latest_model_regst_) {
          if (ModelSatisfySSP(cur_regst, latest_model_regst_)) {
            readable_regsts_.clear();
            readable_regsts_.emplace(in_regst_desc_id_, cur_regst);
            readable_regsts_.emplace(model_regst_desc_id_, latest_model_regst_);
            readable_regsts_.emplace(initial_hidden_regst_desc_id_,
                                     initial_hidden_regsts_.front());
            model_regst2cnt_[latest_model_regst_] += 1;  // insert or add
            return true;
          }
        }
        return false;
      }
    } else {
      auto out_it = pid2out_regst_.find(kv.first);
      if (out_it == pid2out_regst_.end()) {
        continue;
      } else {
        CHECK(cur_pst.IsNextColOf(out_it->second->piece_status()));
        auto model_it = pid2model_regst_.find(cur_pid);
        CHECK(pid2model_regst_.end() != model_it);

        readable_regsts_.clear();
        readable_regsts_.emplace(in_regst_desc_id_, cur_regst);
        readable_regsts_.emplace(model_regst_desc_id_, model_it->second);
        readable_regsts_.emplace(out_regst_desc_id_, out_it->second);
        return true;
      }
    }
  }
  return false;
}

int RnnFwDataCompActor::WaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLER(&RnnFwDataCompActor::HandlerNormal);
  return 0;
}

void RnnFwDataCompActor::AsyncSendMsgToModelProducer() {
  AsyncSendRegstMsgToProducer(latest_model_regst_);
  latest_model_regst_ = nullptr;
}

int RnnFwDataCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
    if (msg_handler() == &RnnFwDataCompActor::HandlerZombie
        || msg_handler() == nullptr) {
      CHECK(pid2in_regsts_.empty());
      CHECK(pid2out_regst_.empty());
      CHECK(initial_hidden_regsts_.empty());
      CHECK(pid2model_regst_.empty());
      CHECK(model_regst2cnt_.empty());
      AsyncSendMsgToModelProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      const PieceStatus& cur_pst = cur_regst->piece_status();
      int64_t cur_pid = cur_pst.piece_id();
      int64_t cur_col_id = cur_pst.col_id();
      int64_t cur_model_vid = cur_regst->model_version_id();

      if (cur_regst_desc_id == in_regst_desc_id_) {
        if (pid2in_regsts_.empty()) { mut_num_of_read_empty() = 0; }
        pid2in_regsts_[cur_pid].push(cur_regst);  // insert or push
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        CHECK_EQ(cur_pid, expected_initial_hidden_piece_id_);
        initial_hidden_regsts_.push(cur_regst);
        expected_initial_hidden_piece_id_ += 1;
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        CHECK_EQ(cur_model_vid, expected_model_version_id_);
        auto iter = model_regst2cnt_.find(latest_model_regst_);
        if (iter == model_regst2cnt_.end()) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        }
        latest_model_regst_ = cur_regst;
        expected_model_version_id_ += 1;
      } else {
        CHECK_EQ(out_regst_desc_id_, cur_regst_desc_id);
        CHECK_EQ(-1, cur_regst->recurrent_flag());
        CHECK(pid2out_regst_.emplace(cur_pid, cur_regst).second);
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int RnnFwDataCompActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  if (TryUpdtStateAsProducedRegst(msg.regst()) == 1) {
    CHECK_EQ(-1, msg.regst()->recurrent_flag());
  }
  ActUntilFail();
  CHECK_NE(-1, in_regst_desc_id_);
  if (pid2in_regsts_.empty()) {
    AsyncSendMsgToModelProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&RnnFwDataCompActor::HandlerZombie);
  }
  return 0;
}

void RnnFwDataCompActor::UpdtInAndModelStatesOfRnnCell() {
  const PieceStatus& cur_pst =
      readable_regsts_.at(in_regst_desc_id_)->piece_status();
  int64_t cur_pid = cur_pst.piece_id();

  if (cur_pst.IsLastCol()) {
    pid2in_regsts_.erase(cur_pid);

    pid2model_regst_.erase(cur_pid);
    Regst* model_regst = readable_regsts_.at(model_regst_desc_id_);
    model_regst2cnt_.at(model_regst) -= 1;
    if (model_regst2cnt_.at(model_regst) == 0) {
      model_regst2cnt_.erase(model_regst);
      if (model_regst != latest_model_regst_) {
        AsyncSendRegstMsgToProducer(model_regst);
      }
    }
  } else {
    pid2in_regsts_.at(cur_pid).pop();
    if (pid2in_regsts_.at(cur_pid).empty()) { pid2in_regsts_.erase(cur_pid); }
  }
  mut_num_of_read_empty() = pid2in_regsts_.empty();
}

void RnnFwDataCompActor::Act() {
  if (in_regst_desc_id_ == -1) {
    AsyncLaunchKernel(kernel_ctx_, [this](int64_t regst_desc_id) -> Regst* {
      TODO();
      return nullptr;
    });
  } else {
    AsyncLaunchKernel(kernel_ctx_, [this](int64_t regst_desc_id) -> Regst* {
      TODO();
      return nullptr;
      // requirement:
      // 1. decide which regst for same regst_desc_id(out_produce/consume)
      // 2. decide which regst is ready for act of different piece id
    });
  }

  int64_t model_version_id = -1;
  PieceStatus tmp_piece_status = ordered_piece_status_;
  if (in_regst_desc_id_ != -1) {
    model_version_id =
        readable_regsts_.at(model_regst_desc_id_)->model_version_id();
    tmp_piece_status = readable_regsts_.at(in_regst_desc_id_)->piece_status();
  }
  AsyncSendRegstMsgToConsumer(
      [this, model_version_id, tmp_piece_status](Regst* regst) {
        regst->set_piece_status(tmp_piece_status);
        regst->set_model_version_id(model_version_id);
        regst->set_is_forward(true);
      });

  if (in_regst_desc_id_ != -1) {
    for (auto& kv : readable_regsts_) {
      if (kv.first == model_regst_desc_id_) { continue; }
      AsyncSendRegstMsgToProducer(kv.second);
    }

    if (readable_regsts_.find(out_regst_desc_id_) == readable_regsts_.end()) {
      initial_hidden_regsts_.pop();
      UpdtInAndModelStatesOfRnnCell();
    } else {
      int64_t piece_id =
          readable_regsts_.at(in_regst_desc_id_)->piece_status().piece_id();
      pid2out_regst_.erase(piece_id);
      UpdtInAndModelStatesOfRnnCell();
    }
  } else {
    if (ordered_piece_status_.GetIntoNextStatus() == -1) {
      in_regst_desc_id_ = -2;
      AsyncSendEORDMsgForAllProducedRegstDesc();
      TrySwitchToZombie();
    }
  }
}

REGISTER_ACTOR(kRnnDataCompTask, true, RnnFwDataCompActor);

}  // namespace oneflow
