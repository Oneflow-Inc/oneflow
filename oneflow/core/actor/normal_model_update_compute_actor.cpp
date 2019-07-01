#include "oneflow/core/actor/normal_model_update_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void NormalMdUpdtCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  const_model_regst_desc_id_ = Name2SoleRegstDescId("const_model");
  int64_t forward_model_regst_desc_id = Name2SoleRegstDescId("forward_model");
  if (forward_model_regst_desc_id != -1) {
    forward_model_regst_ = GetNaiveCurWriteable(forward_model_regst_desc_id);
  } else {
    forward_model_regst_ = nullptr;
  }
  init_remaining_cnt_ = 0;
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (const_model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  next_model_version_id_ = 0;
  for (int64_t model_save_related_actor_id : task_proto.related_save_model_task_ids()) {
    related_save_model_actor_ids_.insert(model_save_related_actor_id);
  }
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  pre_model_regst_ = nullptr;
  const_model_regst_ = nullptr;
  if (const_model_regst_desc_id_ != -1) {
    const_model_regst_ = GetSoleProducedRegst4RegstDescId(const_model_regst_desc_id_);
  }
  send_const_model_regst_ = false;
  OF_SET_MSG_HANDLER(&NormalMdUpdtCompActor::HandlerInitModelAndConstModel);
}

bool NormalMdUpdtCompActor::IsCustomizedWriteReady() const {
  if (const_model_regst_desc_id_ != -1) { CHECK(send_const_model_regst_); }
  return true;
}

void NormalMdUpdtCompActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(const_model_regst_, regst);
  send_const_model_regst_ = false;
}

bool NormalMdUpdtCompActor::CheckOutputActId(int64_t regst_desc_id) const {
  return regst_desc_id != model_regst_desc_id_ && regst_desc_id != const_model_regst_desc_id_;
}

void NormalMdUpdtCompActor::SendConstModelRegstToConsumer() {
  if (const_model_regst_desc_id_ == -1) { return; }
  const_model_regst_->set_model_version_id(next_model_version_id_);
  CHECK_EQ(0, ReadingCnt4ProducedRegst(const_model_regst_));
  const_model_regst_->set_act_id(act_id());
  for (int64_t consumer : const_model_regst_->consumers_actor_id()) {
    AsyncSendMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, const_model_regst_));
  }
  IncreaseReadingCnt4ProducedRegst(const_model_regst_,
                                   const_model_regst_->consumers_actor_id().size());
  IncreaseTotalReadingCnt(const_model_regst_->consumers_actor_id().size());
  send_const_model_regst_ = true;
}

void NormalMdUpdtCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &next_model_version_id_;
  AsyncLaunchKernel(kernel_ctx);
}

void NormalMdUpdtCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* cur_model_regst = GetNaiveCurWriteable(model_regst_desc_id_);
  cur_model_regst->set_model_version_id(next_model_version_id_);
  pre_model_regst_ = cur_model_regst;

  bool need_save_model = NeedModelSave(next_model_version_id_ - 1);
  bool need_send_model = next_model_version_id_ < Global<JobDesc>::Get()->TotalBatchNum();
  HandleProducedNaiveDataRegstToConsumer(
      [cur_model_regst](Regst* regst) { return regst == cur_model_regst; },
      [&](int64_t actor_id) {
        bool is_saving_related =
            related_save_model_actor_ids_.find(actor_id) != related_save_model_actor_ids_.end();
        return (need_save_model && is_saving_related) || (need_send_model && !is_saving_related);
      });
  if (need_save_model && forward_model_regst_ != nullptr) {
    AsyncSendRegstMsgToConsumer(forward_model_regst_);
  }
  next_model_version_id_ += 1;
}

int64_t NormalMdUpdtCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  const auto* job_desc = Global<JobDesc>::Get();
  if (forward_model_regst_ != nullptr && regst_desc_id == forward_model_regst_->regst_desc_id()) {
    return std::min<int64_t>(job_desc->TotalBatchNum() - (forward_model_regst_->act_id() + 1),
                             job_desc->NumOfBatchesInSnapshot());
  }
  return 1;
}

void NormalMdUpdtCompActor::InitRegstBySendToFw(Regst* regst) {
  ActorMsg msg = ActorMsg::BuildRegstMsgToConsumer(actor_id(), related_init_model_actor_id_, regst);
  Global<ActorMsgBus>::Get()->SendMsg(msg);
}

void NormalMdUpdtCompActor::InitModelAndConstBuf() {
  // TODO move the initiation of model and const model from fw op into this function
  if (forward_model_regst_ == nullptr) { return; }
  for (const ExecKernel& ek : exec_kernel_vec()) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    ek.kernel->InitModelAndConstBuf(kernel_ctx, parallel_ctx(),
                                    Global<SnapshotMgr>::Get()->GetReadableSnapshot(),
                                    [&](const std::string& bn_in_op) {
                                      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
                                      Blob* blob = nullptr;
                                      if (forward_model_regst_) {
                                        blob = forward_model_regst_->GetBlobByLbi(lbi);
                                      }
                                      return blob;
                                    });
  }
}

int NormalMdUpdtCompActor::HandlerInitModelAndConstModel(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitModel);
    if (model_regst_desc_id_ != -1) {
      InitRegstBySendToFw(GetNaiveCurWriteable(model_regst_desc_id_));
    }
    if (const_model_regst_desc_id_ != -1) { InitRegstBySendToFw(const_model_regst_); }
    InitModelAndConstBuf();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    init_remaining_cnt_ -= 1;
  } else {
    UNIMPLEMENTED();
  }
  if (init_remaining_cnt_ == 0) {
    OF_SET_MSG_HANDLER(&NormalMdUpdtCompActor::HandlerSendInitialModel);
    Global<RuntimeCtx>::Get()->DecreaseCounter("model_init_cnt");
  }
  return 0;
}

int NormalMdUpdtCompActor::HandlerSendInitialModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  pre_model_regst_ = GetNaiveCurWriteable(model_regst_desc_id_);
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_model_version_id(next_model_version_id_);
    return true;
  });
  SendConstModelRegstToConsumer();
  next_model_version_id_ += 1;
  if (model_regst_desc_id_ != -1) {
    OF_SET_MSG_HANDLER(&NormalMdUpdtCompActor::HandlerNormal);
  } else {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&NormalMdUpdtCompActor::HandlerZombie);
  }
  return 0;
}

REGISTER_ACTOR(TaskType::kNormalMdUpdt, NormalMdUpdtCompActor);

}  // namespace oneflow
