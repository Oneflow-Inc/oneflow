#include "oneflow/core/actor/normal_forward_compute_actor.h"

namespace oneflow {

void NormalForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  const_model_regst_desc_id_ = Name2SoleRegstDescId("const_model");
  const_buf_regst_desc_id_ = Name2SoleRegstDescId("const_buf");
  forward_model_regst_desc_id_ = Name2SoleRegstDescId("forward_model");
  random_seed_ = task_proto.random_seed();
  model_regst_ = nullptr;
  const_model_regst_ = nullptr;
  const_buf_regst_ = nullptr;
  pre_forward_model_regst_ = nullptr;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ = GetNaiveCurWriteable(forward_model_regst_desc_id_);
  }
  if (const_buf_regst_desc_id_ != -1) {
    const_buf_regst_ = GetSoleProducedRegst(const_buf_regst_desc_id_);
  }
  send_const_buf_regst_ = false;
  if (random_seed_ == -1 || (model_regst_desc_id_ == -1 && const_model_regst_desc_id_ == -1)) {
    if (forward_model_regst_desc_id_ != -1 || const_buf_regst_desc_id_ != -1) {
      AsyncInitModelAndConstBuf();
    }
    if (forward_model_regst_desc_id_ != -1) { SendMsgToForwardModelSaveActor(0); }
    if (const_buf_regst_desc_id_ != -1) { SendConstBufInitMsgToBwActor(); }
    OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
  } else {
    OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerInitModelAndConstBuf);
  }
}

std::pair<bool, std::vector<std::string>> NormalForwardCompActor::GetNaiveProducedRegstDescName() {
  HashSet<std::string> naive_names{
      /*kNormalForward*/ "out", "activation", "data_tmp", "fw_buf",
      "forward_model",
      /*kLoss*/ "loss",         "out",        "data_tmp",
      /*kAccuracy*/ "accuracy"};
  return {false, std::vector<std::string>(naive_names.begin(), naive_names.end())};
}

bool NormalForwardCompActor::IsCustomizedWriteReady() {
  if (const_buf_regst_desc_id_ != -1) { CHECK(send_const_buf_regst_); }
  return true;
}

void NormalForwardCompActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(const_buf_regst_, regst);
  const_buf_regst_ = nullptr;
  send_const_buf_regst_ = false;
}

void NormalForwardCompActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  if (const_buf_regst_desc_id_ == -1) { return; }
  CHECK_EQ(0, ReadingCnt4ProducedRegst(const_buf_regst_));
  const_buf_regst_->set_act_id(act_id());
  for (int64_t consumer : const_buf_regst_->consumers_actor_id()) {
    AsyncSendMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, const_buf_regst_));
  }
  IncreaseReadingCnt4ProducedRegst(const_buf_regst_, const_buf_regst_->consumers_actor_id().size());
  IncreaseTotalReadingCnt(const_buf_regst_->consumers_actor_id().size());
  send_const_buf_regst_ = true;
}

bool NormalForwardCompActor::CheckOutputActId(int64_t regst_desc_id) const {
  return regst_desc_id != forward_model_regst_desc_id_;
}

void NormalForwardCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (model_regst_desc_id_ != -1) { handler(model_regst_); }
  if (const_model_regst_desc_id_ != -1) { handler(const_model_regst_); }
}

void NormalForwardCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  if (regst->regst_desc_id() == model_regst_desc_id_) {
    UpdateModelRegstPtr(regst);
  } else if (regst->regst_desc_id() == const_model_regst_desc_id_) {
    CHECK(const_model_regst_ == nullptr);
    const_model_regst_ = regst;
  } else {
    UNIMPLEMENTED();
  }
}

void NormalForwardCompActor::Act() {
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  int64_t piece_id = GetNaiveFirstCurReadable()->piece_id();
  std::tuple<int64_t, std::function<const Blob*(const LogicalBlobId&)>> other_val(
      piece_id, [this](const LogicalBlobId& lbi) -> const Blob* {
        CHECK_NOTNULL(pre_forward_model_regst_);
        return pre_forward_model_regst_->GetBlobByLbi(lbi);
      });
  kernel_ctx.other = &other_val;
  if (forward_model_regst_desc_id_ != -1) {
    pre_forward_model_regst_ = GetNaiveCurWriteable(forward_model_regst_desc_id_);
  }
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == model_regst_desc_id_) {
      return model_regst_;
    } else if (regst_desc_id == const_model_regst_desc_id_) {
      return const_model_regst_;
    } else if (regst_desc_id == const_buf_regst_desc_id_) {
      return const_buf_regst_;
    } else {
      return nullptr;
    }
  });
  AsyncSendNaiveProducedRegstMsgToConsumer([&](Regst* regst) {
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
}

bool NormalForwardCompActor::IsCustomizedReadReady() {
  if (model_regst_desc_id_ != -1 && model_regst_ == nullptr) { return false; }
  if (const_model_regst_desc_id_ != -1 && const_model_regst_ == nullptr) { return false; }
  return true;
}

void NormalForwardCompActor::AsyncReturnAllCustomizedReadableRegst() {
  TryAsyncReturnModelRegst();
  TryAsyncReturnConstModelRegst();
}

int NormalForwardCompActor::HandlerInitModelAndConstBuf(const ActorMsg& msg) {
  CHECK_NE(random_seed_, -1);
  Regst* regst = msg.regst();
  if (regst->regst_desc_id() == model_regst_desc_id_) {
    model_regst_ = regst;
  } else if (regst->regst_desc_id() == const_model_regst_desc_id_) {
    const_model_regst_ = regst;
  } else {
    UNIMPLEMENTED();
  }
  if (model_regst_desc_id_ != -1 && model_regst_ == nullptr) { return 0; }
  if (const_model_regst_desc_id_ != -1 && const_model_regst_ == nullptr) { return 0; }
  AsyncInitModelAndConstBuf();
  if (model_regst_) {
    AsyncSendRegstMsgToProducer(model_regst_);
    model_regst_ = nullptr;
  }
  if (const_model_regst_) {
    AsyncSendRegstMsgToProducer(const_model_regst_);
    const_model_regst_ = nullptr;
  }
  if (forward_model_regst_desc_id_ != -1) { SendMsgToForwardModelSaveActor(0); }
  if (const_buf_regst_desc_id_ != -1) { SendConstBufInitMsgToBwActor(); }
  OF_SET_MSG_HANDLER(&NormalForwardCompActor::HandlerNormal);
  return 0;
}

void NormalForwardCompActor::UpdateModelRegstPtr(Regst* regst) {
  TryAsyncReturnModelRegst();
  model_regst_ = regst;
}

void NormalForwardCompActor::AsyncInitModelAndConstBuf() {
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    std::mt19937 random_seed_gen(random_seed_);
    kernel_ctx.other = &random_seed_gen;
    exec_kernel.kernel->InitModelAndConstBuf(
        kernel_ctx, parallel_ctx(), Global<SnapshotMgr>::Get()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const LogicalBlobId& lbi = exec_kernel.kernel->BnInOp2Lbi(bn_in_op);
          Blob* blob = nullptr;
          if (model_regst_) { blob = model_regst_->GetBlobByLbi(lbi); }
          if (blob == nullptr && const_model_regst_) {
            blob = const_model_regst_->GetBlobByLbi(lbi);
          }
          if (blob == nullptr && const_buf_regst_) { blob = const_buf_regst_->GetBlobByLbi(lbi); }
          if (blob == nullptr && forward_model_regst_desc_id_ != -1) {
            blob = GetNaiveCurWriteable(forward_model_regst_desc_id_)->GetBlobByLbi(lbi);
          }
          return blob;
        });
  }
}

void NormalForwardCompActor::AsyncReturnModelRegst() {
  CHECK_NOTNULL(model_regst_);
  AsyncSendRegstMsgToProducer(model_regst_);
  model_regst_ = nullptr;
}

void NormalForwardCompActor::TryAsyncReturnModelRegst() {
  if (model_regst_) { AsyncReturnModelRegst(); }
}

void NormalForwardCompActor::TryAsyncReturnConstModelRegst() {
  if (const_model_regst_) {
    AsyncSendRegstMsgToProducer(const_model_regst_);
    const_model_regst_ = nullptr;
  }
}

void NormalForwardCompActor::TrySendMsgToForwardModelSaveActor(int64_t piece_id) {
  if (forward_model_regst_desc_id_ == -1) { return; }
  bool is_last_piece_in_batch = (piece_id + 1) % Global<JobDesc>::Get()->NumOfPiecesInBatch() == 0;
  int64_t batch_id = piece_id / Global<JobDesc>::Get()->NumOfPiecesInBatch();
  if (is_last_piece_in_batch && NeedModelSave(batch_id)) {
    SendMsgToForwardModelSaveActor(batch_id);
  }
}

void NormalForwardCompActor::SendMsgToForwardModelSaveActor(int64_t batch_id) {
  AsyncSendNaiveProducedRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(batch_id);
    return regst->regst_desc_id() == forward_model_regst_desc_id_;
  });
}

void NormalForwardCompActor::SendConstBufInitMsgToBwActor() {
  AsyncSendCustomizedProducedRegstMsgToConsumer();
}

REGISTER_ACTOR(TaskType::kNormalForward, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kLoss, NormalForwardCompActor);
REGISTER_ACTOR(TaskType::kAccuracy, NormalForwardCompActor);

}  // namespace oneflow
