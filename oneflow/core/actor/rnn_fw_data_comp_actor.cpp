#include "oneflow/core/actor/rnn_fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void RnnFwDataCompActor::Init(const TaskProto& task_proto,
                              const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);

  in_regst_desc_id_ = RegstDescId4Name("in");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  model_regst_desc_id_ = RegstDescId4Name("model");
  out_regst_desc_id_ = RegstDescId4Name("out");

  expected_model_version_id_ = 0;
  expected_initial_hidden_piece_id_ = 0;
  latest_model_regst_ = nullptr;
  model_tmp_regst_ = nullptr;

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
    CHECK_EQ(-1, model_tmp_regst_desc_id_);
    OF_SET_MSG_HANDLER(&RnnFwDataCompActor::WaitToStart);
  } else {
    // not consider out_regst, else will cause deadlock
    set_num_of_remaining_eord(1 + (initial_hidden_regst_desc_id_ != -1)
                              + (model_regst_desc_id_ != -1)
                              + (model_tmp_regst_desc_id_ != -1));
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

void RnnFwDataCompActor::set_material4act(Material4Act::RnnKernelType type,
                                          Regst* in_regst, Regst* model_regst,
                                          Regst* model_tmp_regst,
                                          Regst* initial_regst,
                                          Regst* out_regst) {
  material4act_.regst_id2regst.clear();

  material4act_.rnn_kernel_type = type;
  if (type == Material4Act::RnnKernelType::kDataLoader) { return; }
  if (type == Material4Act::RnnKernelType::kRnnCellWithInitial) {
    material4act_.regst_id2regst.emplace(initial_hidden_regst_desc_id_,
                                         initial_regst);
  } else if (type == Material4Act::RnnKernelType::kRnnCellWithoutInitial) {
    material4act_.regst_id2regst.emplace(out_regst_desc_id_, out_regst);
  }
  material4act_.regst_id2regst.emplace(in_regst_desc_id_, in_regst);
  if (model_regst_desc_id_ != -1) {
    material4act_.regst_id2regst.emplace(model_regst_desc_id_, model_regst);
  }
  if (model_tmp_regst_desc_id_ != -1) {
    material4act_.regst_id2regst.emplace(model_tmp_regst_desc_id_, model_regst);
  }
}

bool RnnFwDataCompActor::IsReadReady() {
  // DataLoader
  if (in_regst_desc_id_ == -1) {
    set_material4act(Material4Act::RnnKernelType::kDataLoader, nullptr, nullptr,
                     nullptr, nullptr, nullptr);
    return true;
  }
  if (in_regst_desc_id_ == -2) {
    return false;
  }  // dataloader but read to the end

  if (pid2in_regsts_.empty()) { return false; }

  // NormalActor
  if (initial_hidden_regst_desc_id_ == -1) {
    if ((model_regst_desc_id_ != -1 && !latest_model_regst_)
        || (model_tmp_regst_desc_id_ != -1 && !model_tmp_regst_)) {
      return false;
    }
    Regst* cur_regst = pid2in_regsts_.begin()->second.front();
    if (model_regst_desc_id_ != -1) {
      if (ModelSatisfySSP(cur_regst, latest_model_regst_)) {
        set_material4act(Material4Act::RnnKernelType::kNormal, cur_regst,
                         latest_model_regst_, model_tmp_regst_, nullptr,
                         nullptr);
        return true;
      } else {
        return false;
      }
    } else {
      set_material4act(Material4Act::RnnKernelType::kNormal, cur_regst, nullptr,
                       model_tmp_regst_, nullptr, nullptr);
      return true;
    }
  }

  // must handle rnn_cell now
  CHECK_EQ(-1, model_tmp_regst_desc_id_);
  CHECK_NE(-1, model_regst_desc_id_);
  for (const auto& kv : pid2in_regsts_) {  // increasing order by piece_id
    Regst* cur_regst = kv.second.front();
    if (cur_regst->piece_status().col_id() == 0) {
      if (initial_hidden_regsts_.empty()) {
        return false;
      } else {  // if not empty, the front of queue must the same pid
        CHECK(initial_hidden_regsts_.front()->piece_status().piece_id()
              == cur_regst->piece_status().piece_id());
        if (latest_model_regst_) {
          bool ret = ModelSatisfySSP(cur_regst, latest_model_regst_);
          if (ret == true) {
            set_material4act(Material4Act::RnnKernelType::kRnnCellWithInitial,
                             cur_regst, latest_model_regst_, nullptr,
                             initial_hidden_regsts_.front(), nullptr);
          }
          return ret;
        } else {
          return false;
        }
      }
    } else {
      auto iter = pid2out_regst_.find(kv.first);
      if (iter == pid2out_regst_.end()) {
        continue;
      } else {
        CHECK(cur_regst->piece_status().IsNextColOf(
            iter->second->piece_status()));
        auto model_iter =
            pid2model_regst_.find(cur_regst->piece_status().piece_id());
        CHECK(pid2model_regst_.end() != model_iter);
        set_material4act(Material4Act::RnnKernelType::kRnnCellWithoutInitial,
                         cur_regst, model_iter->second, nullptr, nullptr,
                         iter->second);
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

void RnnFwDataCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
  if (model_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(latest_model_regst_);
    latest_model_regst_ = nullptr;
  }
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
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
      AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t cur_piece_id = regst->piece_status().piece_id();
      if (regst->regst_desc_id() == in_regst_desc_id_) {
        CHECK(regst->piece_status() == expected_piece_status_);
        if (pid2in_regsts_.empty()) { mut_num_of_read_empty() = 0; }
        auto it = pid2in_regsts_.find(cur_piece_id);
        if (it == pid2in_regsts_.end()) {
          pid2in_regsts_.emplace(regst->piece_status().piece_id(),
                                 std::queue<Regst*>(std::deque<Regst*>{regst}));
        } else {
          it->second.push(regst);
        }
        expected_piece_status_.GetIntoNextStatus();
      } else if (regst->regst_desc_id() == initial_hidden_regst_desc_id_) {
        CHECK_EQ(regst->piece_status().piece_id(),
                 expected_initial_hidden_piece_id_);
        initial_hidden_regsts_.push(regst);
        expected_initial_hidden_piece_id_ += 1;
      } else if (regst->regst_desc_id() == model_regst_desc_id_) {
        CHECK_EQ(regst->model_version_id(), expected_model_version_id_);
        auto iter = model_regst2cnt_.find(latest_model_regst_);
        if (iter == model_regst2cnt_.end()) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        }
        latest_model_regst_ = regst;
        expected_model_version_id_ += 1;
      } else if (regst->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = regst;
      } else {
        CHECK_EQ(out_regst_desc_id_, regst->regst_desc_id());
        CHECK_EQ(-1, regst->recurrent_flag());
        pid2out_regst_.emplace(cur_piece_id,
                               regst);  // mustn't exist in pid2out_regst_
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
    AsyncSendMsgToModelAndModelTmpProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&RnnFwDataCompActor::HandlerZombie);
  }
  return 0;
}

void RnnFwDataCompActor::UpdtInAndModelStatesOfRnnCell() {
  const PieceStatus& piece_status =
      material4act_.regst_id2regst.at(in_regst_desc_id_)->piece_status();
  if (piece_status.IsLastCol()) {
    pid2in_regsts_.erase(piece_status.piece_id());
    mut_num_of_read_empty() = pid2in_regsts_.empty();

    pid2model_regst_.erase(piece_status.piece_id());
    Regst* model_regst = material4act_.regst_id2regst.at(model_regst_desc_id_);
    auto iter = model_regst2cnt_.find(model_regst);
    if (iter->second == 1) {
      model_regst2cnt_.erase(iter);
      if (model_regst != latest_model_regst_) {
        AsyncSendRegstMsgToProducer(model_regst);
      }
    } else {
      iter->second -= 1;
    }
  } else {
    pid2in_regsts_.at(piece_status.piece_id()).pop();
    if (pid2in_regsts_.at(piece_status.piece_id()).empty()) {
      pid2in_regsts_.erase(piece_status.piece_id());
      mut_num_of_read_empty() = pid2in_regsts_.empty();
    }
  }
}

void RnnFwDataCompActor::Act() {
  if (material4act_.rnn_kernel_type
      == Material4Act::RnnKernelType::kDataLoader) {
    AsyncLaunchKernel(kernel_ctx_,
                      [this](int64_t regst_desc_id) -> Regst* { TODO(); });
  } else if (material4act_.rnn_kernel_type
             == Material4Act::RnnKernelType::kNormal) {
    AsyncLaunchKernel(kernel_ctx_,
                      [this](int64_t regst_desc_id) -> Regst* { TODO(); });
  } else {
    AsyncLaunchKernel(kernel_ctx_, [this](int64_t regst_desc_id) -> Regst* {
      TODO();
      // requirement:
      // 1. decide which regst for same regst_desc_id(out_produce/consume)
      // 2. decide which regst is ready for act of different piece id(use
      // material4act_)
      //
      //
      // Regst* regst = GetCurWriteableRegst(regst_desc_id);
      // if (regst == nullptr) {
      //   return readable_regst_.at(regst_desc_id);
      // } else {
      //   return regst;
      // }
    });
  }

  int64_t model_version_id = -1;
  if (model_regst_desc_id_ != -1) {
    model_version_id = latest_model_regst_->model_version_id();
  }
  PieceStatus tmp_piece_status = ordered_piece_status_;
  if (in_regst_desc_id_ != -1) {
    tmp_piece_status =
        material4act_.regst_id2regst.at(in_regst_desc_id_)->piece_status();
  }
  AsyncSendRegstMsgToConsumer(
      [this, model_version_id, tmp_piece_status](Regst* regst) {
        regst->set_piece_status(tmp_piece_status);
        regst->set_model_version_id(model_version_id);
      });
  int ret = ordered_piece_status_.GetIntoNextStatus();

  if (material4act_.rnn_kernel_type
      != Material4Act::RnnKernelType::kDataLoader) {
    for (auto& kv : material4act_.regst_id2regst) {
      if (kv.first == model_regst_desc_id_) { continue; }
      if (kv.first == model_tmp_regst_desc_id_) { continue; }
      AsyncSendRegstMsgToProducer(kv.second);
    }

    if (material4act_.rnn_kernel_type == Material4Act::RnnKernelType::kNormal) {
      pid2in_regsts_.begin()->second.pop();
      if (pid2in_regsts_.begin()->second.empty()) {
        pid2in_regsts_.erase(pid2in_regsts_.begin());
        mut_num_of_read_empty() = pid2in_regsts_.empty();
      }
    } else if (material4act_.rnn_kernel_type
               == Material4Act::RnnKernelType::kRnnCellWithInitial) {
      initial_hidden_regsts_.pop();
      UpdtInAndModelStatesOfRnnCell();
    } else {  // for kRnnCellWithoutInitial
      int64_t piece_id = material4act_.regst_id2regst.at(in_regst_desc_id_)
                             ->piece_status()
                             .piece_id();
      pid2out_regst_.erase(piece_id);
      UpdtInAndModelStatesOfRnnCell();
    }
  }

  if ((material4act_.rnn_kernel_type
       == Material4Act::RnnKernelType::kDataLoader)
      && (ret == -1)) {  // only for dataloder Actor
    in_regst_desc_id_ = -2;
    AsyncSendMsgToModelAndModelTmpProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    TrySwitchToZombie();
  }
}

REGISTER_ACTOR(kDataCompTask, true, RnnFwDataCompActor);

}  // namespace oneflow
