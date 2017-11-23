#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

void BackwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  out_regst_desc_id_ = RegstDescId4Name("out");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  is_out_diff_eord_ = false;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regsts_[pair.second] = {};
  }
  readable_regst_cnt_ = 0;
  OF_SET_MSG_HANDLER(&BackwardCompActor::HandlerNormal);
}

int BackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id_) {
      is_out_diff_eord_ = true;
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t regst_desc_id = regst->regst_desc_id();
      std::queue<Regst*>& rq = readable_regsts_.at(regst_desc_id);
      if (regst_desc_id == model_tmp_regst_desc_id_) { CHECK(rq.empty()); }
      if (rq.empty()) { readable_regst_cnt_ += 1; }
      rq.push(regst);
      if (regst_desc_id == out_regst_desc_id_ && rq.size() == 1) {
        AsyncReturnModelRegstUntilMatchCurOutRegst();
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool BackwardCompActor::IsReadReady() {
  return readable_regsts_.size() == readable_regst_cnt_;
}

bool BackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_ && readable_regsts_.at(out_regst_desc_id_).empty();
}

void BackwardCompActor::AsyncReturnAllReadableRegst() {
  for (auto& pair : readable_regsts_) {
    while (pair.second.empty() == false) {
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop();
    }
  }
  readable_regst_cnt_ = 0;
}

void BackwardCompActor::AsyncReturnModelRegstUntilMatchCurOutRegst() {
  if (model_regst_desc_id_ == -1) { return; }
  const Regst* cur_out_regst = readable_regsts_.at(out_regst_desc_id_).front();
  int64_t cur_model_id = cur_out_regst->model_version_id();
  std::queue<Regst*>& model_rq = readable_regsts_.at(model_regst_desc_id_);
  while (!model_rq.empty()
         && model_rq.front()->model_version_id() < cur_model_id) {
    AsyncSendRegstMsgToProducer(model_rq.front());
    model_rq.pop();
  }
  if (!model_rq.empty()) {
    CHECK_EQ(model_rq.front()->model_version_id(), cur_model_id);
  }
}

void BackwardCompActor::AsyncReturnModelRegstUntilLastPieceIdGreaterThan(
    int64_t piece_id) {
  std::queue<Regst*>& model_rq = readable_regsts_.at(model_regst_desc_id_);
  while (true) {
    int64_t model_id = model_rq.front()->model_version_id();
    int64_t last_piece_id = GetLastPieceIdForModelVersionId(model_id);
    if (last_piece_id > piece_id) { return; }
    AsyncSendRegstMsgToProducer(model_rq.front());
    model_rq.pop();
  }
}

void BackwardCompActor::Act() {
  std::queue<Regst*>& out_rq = readable_regsts_.at(out_regst_desc_id_);
  int64_t piece_id = out_rq.front()->piece_id();
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regsts_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { regst->set_piece_id(piece_id); });
  AsyncSendRegstMsgToProducer(out_rq.front());
  out_rq.pop();
  if (out_rq.empty()) {
    AsyncReturnModelRegstUntilLastPieceIdGreaterThan(piece_id);
  } else {
    AsyncReturnModelRegstUntilMatchCurOutRegst();
  }
  for (auto& pair : readable_regsts_) {
    if (pair.first == model_tmp_regst_desc_id_) { continue; }
    if (pair.first == model_regst_desc_id_) { continue; }
    if (pair.first == out_regst_desc_id_) { continue; }
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
  }
}

REGISTER_ACTOR(TaskType::kBackward, BackwardCompActor);

}  // namespace oneflow
