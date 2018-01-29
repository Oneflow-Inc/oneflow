#include "oneflow/core/actor/model_update_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void MdUpdtCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  init_remaining_cnt_ = 0;
  if (model_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  if (model_tmp_regst_desc_id_ != -1) { init_remaining_cnt_ += 1; }
  is_model_diff_acc_eord_ = false;
  next_model_version_id_ = 0;
  related_save_model_actor_id_ = task_proto.related_save_model_task_id();
  related_init_model_actor_id_ = task_proto.related_init_model_task_id();
  pre_model_regst_ = nullptr;
  for (const auto& kv : task_proto.consumed_regst_desc_id()) {
    model_diff_acc_regsts_.emplace(kv.second, std::queue<Regst*>());
  }
  readable_model_diff_acc_cnt_ = 0;
  OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerInitModelAndModelTmp);
}

void MdUpdtCompActor::InitRegstBySendToFw(int64_t regst_desc_id) {
  if (regst_desc_id == -1) { return; }
  Regst* regst = GetCurWriteableRegst(regst_desc_id);
  ActorMsg msg = ActorMsg::BuildRegstMsgToConsumer(
      actor_id(), related_init_model_actor_id_, regst);
  ActorMsgBus::Singleton()->SendMsg(msg);
}

int MdUpdtCompActor::HandlerInitModelAndModelTmp(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitModel);
    InitRegstBySendToFw(model_regst_desc_id_);
    InitRegstBySendToFw(model_tmp_regst_desc_id_);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    init_remaining_cnt_ -= 1;
  } else {
    UNEXPECTED_RUN();
  }
  if (init_remaining_cnt_ == 0) {
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerSendInitialModel);
    RuntimeCtx::Singleton()->DecreaseCounter("model_init_cnt");
  }
  return 0;
}

int MdUpdtCompActor::HandlerSendInitialModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  pre_model_regst_ = GetCurWriteableRegst(model_regst_desc_id_);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(next_model_version_id_);
    return true;
  });
  next_model_version_id_ += 1;
  if (JobDesc::Singleton()->IsTrain()) {
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerNormal);
  } else {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerZombie);
  }
  return 0;
}

int MdUpdtCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kEordMsg) {
    is_model_diff_acc_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = actor_msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      int64_t regst_desc_id = regst->regst_desc_id();
      if (model_diff_acc_regsts_.at(regst_desc_id).empty()) {
        readable_model_diff_acc_cnt_ += 1;
      }
      model_diff_acc_regsts_.at(regst->regst_desc_id()).push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void MdUpdtCompActor::Act() {
  Regst* cur_model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  cur_model_regst->set_model_version_id(next_model_version_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  std::tuple<int64_t, const Blob*> other_val(next_model_version_id_,
                                             pre_model_regst_->packed_blob());
  kernel_ctx.other = &other_val;
  pre_model_regst_ = cur_model_regst;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return model_diff_acc_regsts_.at(regst_desc_id).front();
    } else {
      return regst;
    }
  });
  for (auto& kv : model_diff_acc_regsts_) {
    AsyncSendRegstMsgToProducer(kv.second.front());
    kv.second.pop();
    if (kv.second.empty()) { readable_model_diff_acc_cnt_ -= 1; }
  }
  const JobDesc* job_desc = JobDesc::Singleton();
  auto RegstPreProcess = [&](Regst* regst) { return regst == cur_model_regst; };
  if (next_model_version_id_ == job_desc->TotalBatchNum()) {
    AsyncSendRegstMsgToConsumer(RegstPreProcess, [this](int64_t actor_id) {
      return actor_id == related_save_model_actor_id_;
    });
  } else {
    if (next_model_version_id_ % job_desc->NumOfBatchesInSnapshot() == 0) {
      AsyncSendRegstMsgToConsumer(RegstPreProcess);
    } else {
      AsyncSendRegstMsgToConsumer(RegstPreProcess, [this](int64_t actor_id) {
        return actor_id != related_save_model_actor_id_;
      });
    }
  }
  next_model_version_id_ += 1;
}

bool MdUpdtCompActor::IsReadReady() {
  return readable_model_diff_acc_cnt_ == model_diff_acc_regsts_.size();
}

bool MdUpdtCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_model_diff_acc_eord_ && readable_model_diff_acc_cnt_ == 0;
}

bool MdUpdtCompActor::IsWriteReady() {
  return GetCurWriteableRegst(model_regst_desc_id_);
}

void MdUpdtCompActor::AsyncReturnAllReadableRegst() {
  CHECK_EQ(0, readable_model_diff_acc_cnt_);
}

void MdUpdtCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  for (const auto& pair : model_diff_acc_regsts_) {
    handler(pair.second.front());
  }
}

REGISTER_ACTOR(TaskType::kMdUpdt, MdUpdtCompActor);

}  // namespace oneflow
