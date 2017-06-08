#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  cur_handle_ = &MdUpdtCompActor::HandleBeforeInitializeModel;
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
}

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  kernel_ctx.cuda_stream = thread_ctx.compute_cuda_stream;
  (this->*cur_handle_)(actor_msg, kernel_ctx);
}

void MdUpdtCompActor::HandleBeforeInitializeModel(
    const ActorMsg& actor_msg,
    const KernelContext& kernel_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kInitializeModel);
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  model_regst->set_model_version_id(0);
  Regst* model_tmp_regst = GetCurWriteableRegst(model_tmp_regst_desc_id_);
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
  cur_handle_ = &MdUpdtCompActor::HandleBeforeSendInitialModel;
}

void MdUpdtCompActor::HandleBeforeSendInitialModel(
    const ActorMsg& actor_msg,
    const KernelContext& kernel_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kSendInitialModel);
  CurWriteDone();
  SetReadOnlyForRegstDescId(model_tmp_regst_desc_id_);
  cur_handle_ = &MdUpdtCompActor::HandleForUpdateModel;
}

void MdUpdtCompActor::HandleForUpdateModel(
    const ActorMsg&,
    const KernelContext&) {
  TODO();
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
