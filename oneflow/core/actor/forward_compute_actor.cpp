#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

void ForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  in_regst_desc_id_ = RegstDescId4Name("in");
  CHECK_NE(in_regst_desc_id_, -1);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  model_regst_ = nullptr;
  model_tmp_regst_ = nullptr;
  if (model_regst_desc_id_ != -1) {
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerInitModel);
  } else {
    SwitchToHandlerInitModelTmpOrNormal();
  }
}

void ForwardCompActor::SwitchToHandlerInitModelTmpOrNormal() {
  if (model_tmp_regst_desc_id_ != -1) {
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerInitModelTmp);
  } else {
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerNormal);
  }
}

int ForwardCompActor::HandlerInitModel(const ActorMsg& msg) {
  Regst* model_regst = msg.regst();
  CHECK_EQ(model_regst->regst_desc_id(), model_regst_desc_id_);
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
  SwitchToHandlerInitModelTmpOrNormal();
  return 0;
}

int ForwardCompActor::HandlerInitModelTmp(const ActorMsg& msg) {
  Regst* model_tmp_regst = msg.regst();
  CHECK_EQ(model_tmp_regst->regst_desc_id(), model_tmp_regst_desc_id_);
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    exec_kernel.kernel->InitModelTmpBlobs(
        GenDefaultKernelCtx(), parallel_ctx(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          return model_tmp_regst->GetBlobByLbn(lbn);
        });
  }
  AsyncSendRegstMsgToProducer(model_tmp_regst);
  OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerNormal);
  return 0;
}

int ForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id_) { is_in_eord_ = true; }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (regst->regst_desc_id() == in_regst_desc_id_) {
      pending_in_regsts_.push(regst);
    } else if (regst->regst_desc_id() == model_regst_desc_id_) {
      UpdateModelRegstPtr(regst);
    } else if (regst->regst_desc_id() == model_tmp_regst_desc_id_) {
      CHECK(!model_tmp_regst_);
      model_tmp_regst_ = regst;
    } else {
      CHECK_EQ(TryUpdtStateAsProducedRegst(regst), 0);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool ForwardCompActor::IsReadReady() {
  if (pending_in_regsts_.empty()) { return false; }
  if (model_regst_desc_id_ != -1 && !model_regst_) { return false; }
  if (model_tmp_regst_desc_id_ != -1 && !model_tmp_regst_) { return false; }
  return true;
}

bool ForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regsts_.empty();
}

void ForwardCompActor::Act() {
  Regst* in_regst = pending_in_regsts_.front();
  pending_in_regsts_.pop();
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* {
                      if (regst_desc_id == in_regst_desc_id_) {
                        return in_regst;
                      } else if (regst_desc_id == model_regst_desc_id_) {
                        return model_regst_;
                      } else if (regst_desc_id == model_tmp_regst_desc_id_) {
                        return model_tmp_regst_;
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

void ForwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(pending_in_regsts_.empty());
  TryAsyncReturnModelRegst();
  TryAsyncReturnModelTmpRegst();
}

void ForwardCompActor::UpdateModelRegstPtr(Regst* regst) {
  TryAsyncReturnModelRegst();
  model_regst_ = regst;
}

void ForwardCompActor::AsyncReturnModelRegst() {
  CHECK_NOTNULL(model_regst_);
  AsyncSendRegstMsgToProducer(model_regst_);
  model_regst_ = nullptr;
}

void ForwardCompActor::TryAsyncReturnModelRegst() {
  if (model_regst_) { AsyncReturnModelRegst(); }
}

void ForwardCompActor::TryAsyncReturnModelTmpRegst() {
  if (model_tmp_regst_) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
}

std::list<std::string> ForwardCompActor::InputActUidsOfCurAct() const {
  TODO();
  return {""};
}

REGISTER_ACTOR(TaskType::kNormalForward, ForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, ForwardCompActor);

}  // namespace oneflow
