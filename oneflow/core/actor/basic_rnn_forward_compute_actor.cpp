#include "oneflow/core/actor/basic_rnn_forward_compute_actor.h"

namespace oneflow {

void BasicRnnForwardCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  is_in_eord_ = false;

  in_regst_desc_id_ = RegstDescId4Name("in");
  CHECK_NE(-1, in_regst_desc_id_);
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  model_regst_desc_id_ = RegstDescId4Name("model");
  out_regst_desc_id_ = RegstDescId4Name("out");
  latest_model_regst_ = nullptr;

  DecreaseRemainingEordCnt();  // not count 'out', else will cause deadlock
  OF_SET_MSG_HANDLER(&BasicRnnForwardCompActor::HandlerInitModel);
}

int BasicRnnForwardCompActor::HandlerInitModel(const ActorMsg& msg) {
  Regst* model_regst = msg.regst();
  CHECK_EQ(model_regst_desc_id_, model_regst->regst_desc_id());
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    exec_kernel.kernel->InitModelBlobs(
        GenDefaultKernelCtx(), parallel_ctx(),
        SnapshotMgr::Singleton()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          return model_regst->GetBlobByLbn(lbn);
        });
  }
  AsyncSendRegstMsgToProducer(model_regst);
  OF_SET_MSG_HANDLER(&BasicRnnForwardCompActor::HandlerNormal);
  return 0;
}

int BasicRnnForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id_) { is_in_eord_ = true; }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      const PieceStatus& cur_pst = cur_regst->piece_status();
      int64_t cur_pid = cur_pst.piece_id();

      if (cur_regst_desc_id == in_regst_desc_id_) {
        pid2in_regsts_[cur_pid].push(cur_regst);  // insert or push
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        initial_hidden_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        auto iter = model_regst2cnt_.find(latest_model_regst_);
        if (iter == model_regst2cnt_.end()) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        }
        latest_model_regst_ = cur_regst;
      } else if (cur_regst_desc_id == out_regst_desc_id_ 
          && cur_regst->recurrent_flag() == -1) {
        CHECK(pid2out_regst_.emplace(cur_pid, cur_regst).second);
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

bool BasicRnnForwardCompActor::IsReadReady() {
  if (pid2in_regsts_.empty() || !latest_model_regst_) { return false; }
  for (const auto& kv : pid2in_regsts_) {  // increasing order by pid
    Regst* cur_regst = kv.second.front();
    const PieceStatus& cur_pst = cur_regst->piece_status();
    int64_t cur_pid = cur_pst.piece_id();
    CHECK_EQ(cur_pid, kv.first);
    int64_t cur_col_id = cur_pst.col_id();

    if (cur_col_id == 0) {
      if (initial_hidden_regsts_.empty()) {
        return false;
      } else {
        CHECK_EQ(cur_pid,
                 initial_hidden_regsts_.front()->piece_status().piece_id());
        int64_t last_pid = GetLastPieceIdForModelVersionId(
            latest_model_regst_->model_version_id());
        if (cur_pid > last_pid) {
          return false;
        } else {
          readable_regsts_.clear();
          readable_regsts_.emplace(in_regst_desc_id_, cur_regst);
          readable_regsts_.emplace(model_regst_desc_id_, latest_model_regst_);
          readable_regsts_.emplace(initial_hidden_regst_desc_id_,
                                   initial_hidden_regsts_.front());

          CHECK(pid2model_regst_.emplace(cur_pid, latest_model_regst_).second);
          model_regst2cnt_[latest_model_regst_] += 1;  // insert or add
          return true;
        }
      }
    } else {
      auto out_it = pid2out_regst_.find(cur_pid);
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

bool BasicRnnForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pid2in_regsts_.empty();
}

void BasicRnnForwardCompActor::UpdtInAndModelStates() {
  const PieceStatus& cur_pst =
      readable_regsts_.at(in_regst_desc_id_)->piece_status();
  int64_t cur_pid = cur_pst.piece_id();

  if (cur_pst.IsLastCol()) {
    pid2in_regsts_.erase(cur_pid);
    pid2model_regst_.erase(cur_pid);

    Regst* model_regst = readable_regsts_.at(model_regst_desc_id_);
    int64_t model_vid = model_regst->model_version_id();
    model_regst2cnt_.at(model_regst) -= 1;

    if (cur_pid == GetLastPieceIdForModelVersionId(model_vid)) {
      if (model_regst2cnt_.at(model_regst) == 0) {
        model_regst2cnt_.erase(model_regst);
        AsyncSendRegstMsgToProducer(model_regst);
        if (model_regst == latest_model_regst_) {
          latest_model_regst_ = nullptr;
        }
      } else {
        models_to_be_released_.insert(model_regst);
      }
    } else {
      if (model_regst2cnt_.at(model_regst) == 0) {
        model_regst2cnt_.erase(model_regst);
        if (models_to_be_released_.find(model_regst)
            == models_to_be_released_.end()) {
          AsyncSendRegstMsgToProducer(model_regst);
          if (latest_model_regst_ == model_regst) {
            latest_model_regst_ = nullptr;
          }
        }
      }
    }
  } else {
    pid2in_regsts_.at(cur_pid).pop();
    if (pid2in_regsts_.at(cur_pid).empty()) { pid2in_regsts_.erase(cur_pid); }
  }
}

void BasicRnnForwardCompActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* { return nullptr; });
  Regst* in_regst = readable_regsts_.at(in_regst_desc_id_);
  Regst* model_regst = readable_regsts_.at(model_regst_desc_id_);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(model_regst->model_version_id());
    regst->set_is_forward(true);
  });

  for (auto& kv : readable_regsts_) {
    if (kv.first == model_regst_desc_id_) { continue; }
    AsyncSendRegstMsgToProducer(kv.second);
  }
  int64_t pid = in_regst->piece_status().piece_id();
  if (readable_regsts_.find(initial_hidden_regst_desc_id_)
      != readable_regsts_.end()) {
    initial_hidden_regsts_.pop();
  } else {
    pid2out_regst_.erase(pid);
  }
  UpdtInAndModelStates();
}

void BasicRnnForwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(pid2out_regst_.empty());
  CHECK(initial_hidden_regsts_.empty());
  CHECK(pid2model_regst_.empty());
  CHECK(model_regst2cnt_.empty());
  CHECK(models_to_be_released_.empty());
  CHECK(pid2out_regst_.empty());
  if (latest_model_regst_) {
    AsyncSendRegstMsgToProducer(latest_model_regst_);
    latest_model_regst_ = nullptr;
  }
}

REGISTER_ACTOR(TaskType::kBasicRnnForward, BasicRnnForwardCompActor);

}  // namespace oneflow
