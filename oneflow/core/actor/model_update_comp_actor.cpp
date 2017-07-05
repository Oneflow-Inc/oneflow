#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  next_model_version_id_ = 0;
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleBeforeInitializeModel);
}

int MdUpdtCompActor::HandleBeforeInitializeModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kInitializeModel);
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  model_regst->set_model_version_id(next_model_version_id_++);
  Regst* model_tmp_regst = GetCurWriteableRegst(model_tmp_regst_desc_id_);
  HashSet<const Kernel*> kernels;
  auto CollectKernelsFromLbn = [&kernels](const std::string& lbn) {
    std::string op_name = GetOpNameFromLbn(lbn);
    kernels.insert(KernelMgr::Singleton()->GetKernelFromOpName(op_name));
  };
  model_regst->ForEachLbn(CollectKernelsFromLbn);
  model_tmp_regst->ForEachLbn(CollectKernelsFromLbn);

  for (const Kernel* kernel : kernels) {
    kernel->InitModelAndModelTmpBlobs(
        GenDefaultKernelCtx(), parallel_policy(), parallel_id(), parallel_num(),
        SnapshotMgr::Singleton()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = kernel->Lbn4BnInOp(bn_in_op);
          Blob* ret = model_regst->GetBlobPtrFromLbn(lbn);
          if (ret == nullptr) { ret = model_tmp_regst->GetBlobPtrFromLbn(lbn); }
          CHECK(ret != nullptr);
          return ret;
        });
  }
  AsyncDo([]() { RuntimeCtx::Singleton()->OneModelInitDone(); });
  OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleBeforeSendInitialModel);
  return 0;
}

int MdUpdtCompActor::HandleBeforeSendInitialModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  AsyncSendReadableRegstMsg();
  SetReadOnlyForRegstDescId(model_tmp_regst_desc_id_);
  AsyncSendEORDMsgToSubscribers(model_tmp_regst_desc_id_);
  if (JobDesc::Singleton()->is_train()) {
    OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleNormal);
  } else {
    AsyncSendEORDMsgToSubscribers(model_regst_desc_id_);
    OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleWaitUntilReadingCntEqualZero);
  }
  return 0;
}

int MdUpdtCompActor::HandleNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleWaitUntilNoReadableRegst);
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_warpper = actor_msg.regst_warpper();
    if (TryUpdtStateAsProducedRegst(regst_warpper->regst_raw_ptr()) != 0) {
      waiting_model_diff_acc_queue_.push(regst_warpper);
    }
    TryActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

int MdUpdtCompActor::HandleWaitUntilNoReadableRegst(const ActorMsg& actor_msg) {
  CHECK_EQ(
      TryUpdtStateAsProducedRegst(actor_msg.regst_warpper()->regst_raw_ptr()),
      0);
  TryActUntilFail();
  if (waiting_model_diff_acc_queue_.empty()) {
    AsyncSendEORDMsgToSubscribers(model_regst_desc_id_);
    if (total_reading_cnt() == 0) {
      OF_SET_MSG_HANDLE(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLE(&MdUpdtCompActor::HandleWaitUntilReadingCntEqualZero);
      return 0;
    }
  }
  return 0;
}

void MdUpdtCompActor::Act() {
  auto model_diff_acc_wpr = waiting_model_diff_acc_queue_.front();
  waiting_model_diff_acc_queue_.pop();
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  auto model_wpr = std::make_shared<LocalRegstWarpper>(model_regst);
  model_regst->set_model_version_id(next_model_version_id_++);
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [&](int64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
        if (regst_desc_id == model_regst_desc_id_) {
          return model_wpr;
        } else {
          return model_diff_acc_wpr;
        }
      });
  AsyncSendReadableRegstMsg();
  AsyncSendRegstMsgToProducer(model_diff_acc_wpr);
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
