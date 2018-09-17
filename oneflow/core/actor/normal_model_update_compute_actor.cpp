#include "oneflow/core/actor/normal_model_update_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void NormalMdUpdtCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  const_model_regst_desc_id_ = Name2SoleRegstDescId("const_model");
  int64_t forward_model_regst_desc_id = Name2SoleRegstDescId("forward_model");
  if (forward_model_regst_desc_id != -1) {
    forward_model_regst_ = GetCurWriteableRegst(forward_model_regst_desc_id);
  } else {
    forward_model_regst_ = nullptr;
  }
  init_remaining_cnt_ = 0;
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (const_model_regst_desc_id_ != -1) {
    init_remaining_cnt_ += 1;
    DecreaseActualWriteableProducedDataRegstDescNum(1);
  }
  next_model_version_id_ = 0;
  for (int64_t model_save_related_actor_id : task_proto.related_save_model_task_ids()) {
    related_save_model_actor_ids_.insert(model_save_related_actor_id);
  }
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  pre_model_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&NormalMdUpdtCompActor::HandlerInitModelAndConstModel);
}

bool NormalMdUpdtCompActor::CheckOutputActId(int64_t regst_desc_id) const {
  return regst_desc_id != model_regst_desc_id_ && regst_desc_id != const_model_regst_desc_id_;
}

void NormalMdUpdtCompActor::Act() {
  Regst* cur_model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  cur_model_regst->set_model_version_id(next_model_version_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &next_model_version_id_;
  pre_model_regst_ = cur_model_regst;
  AsyncLaunchKernel(kernel_ctx);
  const JobDesc* job_desc = Global<JobDesc>::Get();
  auto RegstPreProcess = [&](Regst* regst) { return regst == cur_model_regst; };
  bool need_save_model = NeedModelSave(next_model_version_id_ - 1);
  bool need_send_model = next_model_version_id_ < job_desc->TotalBatchNum();
  AsyncSendRegstMsgToConsumer(RegstPreProcess, [&](int64_t actor_id) {
    bool is_for_save_model =
        related_save_model_actor_ids_.find(actor_id) != related_save_model_actor_ids_.end();
    return (need_save_model && is_for_save_model) || (need_send_model && !is_for_save_model);
  });
  if (need_save_model && forward_model_regst_ != nullptr) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) { return regst == forward_model_regst_; });
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

void NormalMdUpdtCompActor::InitRegstBySendToFw(int64_t regst_desc_id) {
  if (regst_desc_id == -1) { return; }
  Regst* regst = GetCurWriteableRegst(regst_desc_id);
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
    InitRegstBySendToFw(model_regst_desc_id_);
    InitRegstBySendToFw(const_model_regst_desc_id_);
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
  pre_model_regst_ = GetCurWriteableRegst(model_regst_desc_id_);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(next_model_version_id_);
    return true;
  });
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
