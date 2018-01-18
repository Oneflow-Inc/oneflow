#include "oneflow/core/actor/normal_backward_compute_actor.h"

namespace oneflow {

void NormalBackwardCompActor::VirtualBackwardCompActorInit(
    const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regsts_[pair.second] = {};
  }
  readable_regst_cnt_ = 0;
  OF_SET_MSG_HANDLER(&NormalBackwardCompActor::HandlerNormal);
}

int NormalBackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id()) {
      set_is_out_diff_eord(true);
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t regst_desc_id = regst->regst_desc_id();
      std::queue<Regst*>& rq = readable_regsts_.at(regst_desc_id);
      if (regst_desc_id == model_tmp_regst_desc_id()) { CHECK(rq.empty()); }
      if (rq.empty()) { readable_regst_cnt_ += 1; }
      rq.push(regst);
      if (regst_desc_id == out_regst_desc_id() && rq.size() == 1) {
        std::queue<Regst*>& model_rq =
            readable_regsts_.at(model_regst_desc_id());
        AsyncReturnModelRegstUntilMatchCurOutRegst(
            rq.front()->model_version_id(), model_rq);
        if (model_rq.empty()) { readable_regst_cnt_ -= 1; }
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool NormalBackwardCompActor::IsReadReady() {
  return readable_regsts_.size() == readable_regst_cnt_;
}

bool NormalBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord()
         && readable_regsts_.at(out_diff_regst_desc_id()).empty();
}

void NormalBackwardCompActor::AsyncReturnAllReadableRegst() {
  for (auto& pair : readable_regsts_) {
    while (pair.second.empty() == false) {
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop();
    }
  }
  readable_regst_cnt_ = 0;
}

void NormalBackwardCompActor::Act() {
  std::queue<Regst*>& out_rq = readable_regsts_.at(out_regst_desc_id());
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
  Regst* out_diff_regst = readable_regsts_.at(out_diff_regst_desc_id()).front();
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_col_id(out_diff_regst->col_id());
    regst->set_max_col_id(out_diff_regst->max_col_id());
    return true;
  });
  AsyncSendRegstMsgToProducer(out_rq.front());
  out_rq.pop();
  std::queue<Regst*>& model_rq = readable_regsts_.at(model_regst_desc_id());
  if (out_rq.empty()) {
    AsyncReturnModelRegstUntilLastPieceIdGreaterThan(piece_id, model_rq);
    readable_regst_cnt_ -= 1;
  } else {
    AsyncReturnModelRegstUntilMatchCurOutRegst(
        out_rq.front()->model_version_id(), model_rq);
  }
  if (model_rq.empty()) { readable_regst_cnt_ -= 1; }
  for (auto& pair : readable_regsts_) {
    if (pair.first == model_tmp_regst_desc_id()) { continue; }
    if (pair.first == model_regst_desc_id()) { continue; }
    if (pair.first == out_regst_desc_id()) { continue; }
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

REGISTER_ACTOR(TaskType::kNormalBackward, NormalBackwardCompActor);

}  // namespace oneflow
