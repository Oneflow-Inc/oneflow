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

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext& thread_ctx) {
  (this->*cur_msg_handle_)(actor_msg, thread_ctx);
}

void MdUpdtCompActor::HandleBeforeInitKernelCtx(
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
}

void MdUpdtCompActor::HandleBeforeInitializeModel(
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
}

void MdUpdtCompActor::HandleBeforeSendInitialModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kSendInitialModel);
  AsyncSendMsgToRegstReader();
  SetReadOnlyForRegstDescId(model_tmp_regst_desc_id_);
  cur_msg_handle_ = &MdUpdtCompActor::HandleForUpdateModel;
}

void MdUpdtCompActor::HandleForUpdateModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK(actor_msg.actor_cmd() == ActorCmd::kStop);
    TODO();
    cur_msg_handle_ = nullptr;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    ProcessRegstFromMsg(actor_msg.regst_warpper());
  } else {
    UNEXPECTED_RUN();
  }
}

void MdUpdtCompActor::ProcessRegstFromMsg(
    std::shared_ptr<RegstWarpper> regst_warpper) {
  if (TryUpdtStateAsFromRegstReader(regst_warpper->regst_raw_ptr()) != 0) {
    waiting_model_diff_acc_queue_.push(regst_warpper);
  }
  if (!waiting_model_diff_acc_queue_.empty() && IsWriteReady()) {
    auto model_diff_acc_wpr = waiting_model_diff_acc_queue_.front();
    waiting_model_diff_acc_queue_.pop();
    Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
    auto model_wpr = std::make_shared<LocalRegstWarpper>(model_regst);
    model_regst->set_model_version_id(next_model_version_id_++);
    AsyncWardKernelAndSendMsgToRegstReader(
        [&](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      if (regst_desc_id == model_regst_desc_id_) {
        return model_wpr;
      } else {
        return model_diff_acc_wpr;
      }
    });
  }
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
