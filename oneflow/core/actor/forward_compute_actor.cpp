#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

void ForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  in_regst_desc_id_ = RegstDescId4Name("in");
  CHECK_NE(in_regst_desc_id_, -1);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  forward_model_regst_desc_id_ = RegstDescId4Name("forward_model");
  random_seed_ = task_proto.random_seed();
  model_regst_ = nullptr;
  model_tmp_regst_ = nullptr;
  pre_forward_model_regst_ = nullptr;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ =
        GetCurWriteableRegst(forward_model_regst_desc_id_);
  }
  if (random_seed_ == -1
      || (model_regst_desc_id_ == -1 && model_tmp_regst_desc_id_ == -1)) {
    if (forward_model_regst_desc_id_ != -1) {
      AsyncInitModel();
      SendMsgToForwardModelSaveActor(0);
    }
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerNormal);
  } else {
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerInitModelAndModelTmp);
  }
}

int ForwardCompActor::HandlerInitModelAndModelTmp(const ActorMsg& msg) {
  CHECK_NE(random_seed_, -1);
  Regst* regst = msg.regst();
  if (regst->regst_desc_id() == model_regst_desc_id_) {
    model_regst_ = regst;
  } else if (regst->regst_desc_id() == model_tmp_regst_desc_id_) {
    model_tmp_regst_ = regst;
  } else {
    UNIMPLEMENTED();
  }
  if (model_regst_desc_id_ != -1 && model_regst_ == nullptr) { return 0; }
  if (model_tmp_regst_desc_id_ != -1 && model_tmp_regst_ == nullptr) {
    return 0;
  }
  AsyncInitModel();
  if (model_regst_) {
    AsyncSendRegstMsgToProducer(model_regst_);
    model_regst_ = nullptr;
  }
  if (model_tmp_regst_) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
  if (forward_model_regst_desc_id_ != -1) { SendMsgToForwardModelSaveActor(0); }
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
    UNIMPLEMENTED();
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

void ForwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(pending_in_regsts_.empty());
  TryAsyncReturnModelRegst();
  TryAsyncReturnModelTmpRegst();
}

void ForwardCompActor::Act() {
  Regst* in_regst = pending_in_regsts_.front();
  pending_in_regsts_.pop();
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  int64_t piece_id = in_regst->piece_id();
  std::tuple<int64_t, std::function<const Blob*(const std::string&)>> other_val(
      piece_id, [=](const std::string& lbn) -> const Blob* {
        CHECK_NOTNULL(pre_forward_model_regst_);
        return pre_forward_model_regst_->GetBlobByLbn(lbn);
      });
  kernel_ctx.other = &other_val;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ =
        GetCurWriteableRegst(forward_model_regst_desc_id_);
  }
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
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
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
    return regst->regst_desc_id() != forward_model_regst_desc_id_;
  });
  if (Global<JobDesc>::Get()->IsTrain()) {
    if (model_regst_) {
      int64_t last_piece_id = GetLastPieceIdForModelVersionId(model_version_id);
      CHECK_LE(piece_id, last_piece_id);
      if (piece_id == last_piece_id) { AsyncReturnModelRegst(); }
    }
    TrySendMsgToForwardModelSaveActor(piece_id);
  }
  AsyncSendRegstMsgToProducer(in_regst);
}

void ForwardCompActor::UpdateModelRegstPtr(Regst* regst) {
  TryAsyncReturnModelRegst();
  model_regst_ = regst;
}

void ForwardCompActor::AsyncInitModel() {
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    std::mt19937 random_seed_gen(random_seed_);
    kernel_ctx.other = &random_seed_gen;
    exec_kernel.kernel->InitModelAndModelTmp(
        kernel_ctx, parallel_ctx(),
        Global<SnapshotMgr>::Get()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          Blob* blob = nullptr;
          if (model_regst_) { blob = model_regst_->GetBlobByLbn(lbn); }
          if (blob == nullptr && model_tmp_regst_) {
            blob = model_tmp_regst_->GetBlobByLbn(lbn);
          }
          if (blob == nullptr && forward_model_regst_desc_id_ != -1) {
            blob = GetCurWriteableRegst(forward_model_regst_desc_id_)
                       ->GetBlobByLbn(lbn);
          }
          return blob;
        });
  }
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

void ForwardCompActor::TrySendMsgToForwardModelSaveActor(int64_t piece_id) {
  if (forward_model_regst_desc_id_ == -1) { return; }
  bool is_last_piece_in_batch =
      (piece_id + 1) % Global<JobDesc>::Get()->NumOfPiecesInBatch() == 0;
  int64_t batch_id = piece_id / Global<JobDesc>::Get()->NumOfPiecesInBatch();
  if (is_last_piece_in_batch && NeedModelSave(batch_id)) {
    SendMsgToForwardModelSaveActor(batch_id);
  }
}

void ForwardCompActor::SendMsgToForwardModelSaveActor(int64_t batch_id) {
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(batch_id);
    return regst->regst_desc_id() == forward_model_regst_desc_id_;
  });
}

void ForwardCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(pending_in_regsts_.front());
  if (model_regst_desc_id_ != -1) { handler(model_regst_); }
  if (model_tmp_regst_desc_id_ != -1) { handler(model_tmp_regst_); }
}

REGISTER_ACTOR(TaskType::kNormalForward, ForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, ForwardCompActor);

}  // namespace oneflow
