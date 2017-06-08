#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  state_ = &BeforeInitializeModelState::Singleton();
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
}

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  kernel_ctx.cuda_stream = thread_ctx.compute_cuda_stream;
  state_->ProcessMsg(actor_msg, kernel_ctx, this);
}

void MdUpdtCompActor::BeforeInitializeModelState::ProcessMsg(
    const ActorMsg& actor_msg,
    const KernelContext& kernel_ctx,
    MdUpdtCompActor* actor) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kInitializeModel);
  Regst* model_regst = actor->GetCurWriteableRegst(actor->model_regst_desc_id_);
  model_regst->set_model_version_id(0);
  Regst* model_tmp_regst = actor->GetCurWriteableRegst(actor->model_tmp_regst_desc_id_);
  HashSet<const Kernel*> kernels;
  auto CollectKernelsFromLbn = [&kernels](const std::string& lbn) {
    std::string op_name = GetOpNameFromLbn(lbn);
    kernels.insert(KernelMgr::Singleton().GetKernelFromOpName(op_name));
  };
  model_regst->ForEachLbn(CollectKernelsFromLbn);
  model_tmp_regst->ForEachLbn(CollectKernelsFromLbn);
  for (const Kernel* kernel : kernels) {
    kernel->InitModelAndModelTmpBlobs(kernel_ctx,
                                      [&](const std::string& bn_in_op) {
      const std::string& lbn = kernel->Lbn4BnInOp(bn_in_op);
      Blob* ret = model_regst->GetBlobPtrFromLbn(lbn);
      if (ret == nullptr) { ret = model_tmp_regst->GetBlobPtrFromLbn(lbn); }
      CHECK(ret != nullptr);
      return ret;
    });
  }
  actor->state_ = &BeforeSendInitialModelState::Singleton();
}

void MdUpdtCompActor::BeforeSendInitialModelState::ProcessMsg(
    const ActorMsg& actor_msg,
    const KernelContext& kernel_ctx,
    MdUpdtCompActor* actor) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kSendInitialModel);
  actor->CurWriteDone();
  actor->SetReadOnlyForRegstDescId(actor->model_tmp_regst_desc_id_);
  actor->state_ = &UpdateModelState::Singleton();
}

void MdUpdtCompActor::UpdateModelState::ProcessMsg(
    const ActorMsg&,
    const KernelContext&,
    MdUpdtCompActor* actor) {
  TODO();
}

void MdUpdtCompActor::EndState::ProcessMsg(
    const ActorMsg&,
    const KernelContext&,
    MdUpdtCompActor* actor) {
  TODO();
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
