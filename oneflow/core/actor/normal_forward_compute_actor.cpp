#include "oneflow/core/actor/normal_forward_compute_actor.h"

namespace oneflow {

void NormalForwardCompActor::VirtualForwardCompActorInit(
    const TaskProto& task_proto) {
  model_regst_ = nullptr;
  if (model_regst_desc_id() != -1) {
    OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerInitModel);
  } else {
    SwitchToHandlerInitModelTmpOrNormal();
  }
}

void NormalForwardCompActor::SetMsgHandlerOfNormal() {
  OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
}

int NormalForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id()) {
      set_is_in_eord(true);
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (regst->regst_desc_id() == in_regst_desc_id()) {
      pending_in_regsts_.push(regst);
    } else if (regst->regst_desc_id() == model_regst_desc_id()) {
      UpdateModelRegstPtr(regst);
    } else if (regst->regst_desc_id() == model_tmp_regst_desc_id()) {
      CHECK(!model_tmp_regst());
      set_mode_tmp_regst(regst);
    } else {
      CHECK_EQ(TryUpdtStateAsProducedRegst(regst), 0);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool NormalForwardCompActor::IsReadReady() {
  if (pending_in_regsts_.empty()) { return false; }
  if (model_regst_desc_id() != -1 && !model_regst_) { return false; }
  if (model_tmp_regst_desc_id() != -1 && !model_tmp_regst()) { return false; }
  return true;
}

bool NormalForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord() && pending_in_regsts_.empty();
}

void NormalForwardCompActor::Act() {
  Regst* in_regst = pending_in_regsts_.front();
  pending_in_regsts_.pop();
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* {
                      if (regst_desc_id == in_regst_desc_id()) {
                        return in_regst;
                      } else if (regst_desc_id == model_regst_desc_id()) {
                        return model_regst_;
                      } else if (regst_desc_id == model_tmp_regst_desc_id()) {
                        return model_tmp_regst();
                      } else {
                        return GetCurWriteableRegst(regst_desc_id);
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    regst->set_model_version_id(model_version_id);
    return true;
  });
  if (JobDesc::Singleton()->IsTrain() && model_regst_) {
    int64_t last_piece_id = GetLastPieceIdForModelVersionId(model_version_id);
    CHECK_LE(in_regst->piece_id(), last_piece_id);
    if (in_regst->piece_id() == last_piece_id) { AsyncReturnModelRegst(); }
  }
  AsyncSendRegstMsgToProducer(in_regst);
}

void NormalForwardCompActor::UpdateModelRegstPtr(Regst* regst) {
  TryAsyncReturnModelRegst();
  model_regst_ = regst;
}

void NormalForwardCompActor::AsyncReturnModelRegst() {
  CHECK_NOTNULL(model_regst_);
  AsyncSendRegstMsgToProducer(model_regst_);
  model_regst_ = nullptr;
}

void NormalForwardCompActor::TryAsyncReturnModelRegst() {
  if (model_regst_) { AsyncReturnModelRegst(); }
}

void NormalForwardCompActor::CheckBeforeAsyncReturnAllReadableRegst() {
  CHECK(pending_in_regsts_.empty());
}

REGISTER_ACTOR(TaskType::kNormalForward, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, NormalForwardCompActor);

}  // namespace oneflow
