#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  cur_msg_handle_ = &MdUpdtCompActor::HandleBeforeInitKernelCtx;
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  next_model_version_id_ = 0;
}

int MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext& thread_ctx) {
  return (this->*cur_msg_handle_)(actor_msg, thread_ctx);
}

int MdUpdtCompActor::HandleBeforeInitKernelCtx(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kInitKernelCtx);
  if (thread_ctx.cpu_stream) {
    mut_kernel_ctx().reset(new CpuKernelCtx(thread_ctx.cpu_stream));
  } else {
    mut_kernel_ctx().reset(new CudaKernelCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  cur_msg_handle_ = &MdUpdtCompActor::HandleBeforeInitializeModel;
  return 0;
}

int MdUpdtCompActor::HandleBeforeInitializeModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kInitializeModel);
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  model_regst->set_model_version_id(next_model_version_id_++);
  Regst* model_tmp_regst = GetCurWriteableRegst(model_tmp_regst_desc_id_);
  HashSet<const Kernel*> kernels;
  auto CollectKernelsFromLbn = [&kernels](const std::string& lbn) {
    std::string op_name = GetOpNameFromLbn(lbn);
    kernels.insert(KernelMgr::Singleton().GetKernelFromOpName(op_name));
  };
  model_regst->ForEachLbn(CollectKernelsFromLbn);
  model_tmp_regst->ForEachLbn(CollectKernelsFromLbn);
  for (const Kernel* kernel : kernels) {
    kernel->InitModelAndModelTmpBlobs(kernel_ctx(),
                                      [&](const std::string& bn_in_op) {
      const std::string& lbn = kernel->Lbn4BnInOp(bn_in_op);
      Blob* ret = model_regst->GetBlobPtrFromLbn(lbn);
      if (ret == nullptr) { ret = model_tmp_regst->GetBlobPtrFromLbn(lbn); }
      CHECK(ret != nullptr);
      return ret;
    });
  }
  cur_msg_handle_ = &MdUpdtCompActor::HandleBeforeSendInitialModel;
  return 0;
}

int MdUpdtCompActor::HandleBeforeSendInitialModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kSendInitialModel);
  AsyncSendReadableRegstMsg();
  SetReadOnlyForRegstDescId(model_tmp_regst_desc_id_);
  AsyncSendStopMsgToRegstSubscribers(model_tmp_regst_desc_id_);
  cur_msg_handle_ = &MdUpdtCompActor::HandleUpdateModel;
  return 0;
}

int MdUpdtCompActor::HandleUpdateModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK(actor_msg.actor_cmd() == ActorCmd::kStop);
    cur_msg_handle_ = &MdUpdtCompActor::HandleUpdtModelWhenNoReadableRegstMsg;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_warpper = actor_msg.regst_warpper();
    if (TryUpdtStateAsProducedRegst(regst_warpper->regst_raw_ptr()) != 0) {
      waiting_model_diff_acc_queue_.push(regst_warpper);
    }
    TryWardKernelAndSendMsg();
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

int MdUpdtCompActor::HandleUpdtModelWhenNoReadableRegstMsg(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(
      actor_msg.regst_warpper()->regst_raw_ptr()), 0);
  TryWardKernelAndSendMsg();
  if (total_reading_cnt() == 0) {
    cur_msg_handle_ = nullptr;
    return 1;
  }
  return 0;
}

void MdUpdtCompActor::TryWardKernelAndSendMsg() {
  if (!waiting_model_diff_acc_queue_.empty() && IsWriteReady()) {
    auto model_diff_acc_wpr = waiting_model_diff_acc_queue_.front();
    waiting_model_diff_acc_queue_.pop();
    Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
    auto model_wpr = std::make_shared<LocalRegstWarpper>(model_regst);
    model_regst->set_model_version_id(next_model_version_id_++);
    AsyncWardKernel(
        [&](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      if (regst_desc_id == model_regst_desc_id_) {
        return model_wpr;
      } else {
        return model_diff_acc_wpr;
      }
    });
    AsyncSendReadableRegstMsg();
    ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(
        model_diff_acc_wpr->producer_actor_id(),
        model_diff_acc_wpr->regst_raw_ptr());
    AsyncDo([msg]() {
      ActorMsgBus::Singleton().SendMsg(msg);
    });
  }
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
