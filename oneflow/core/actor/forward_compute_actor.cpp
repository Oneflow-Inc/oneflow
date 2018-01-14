#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

void ForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  in_regst_desc_id_ = RegstDescId4Name("in");
  CHECK_NE(in_regst_desc_id_, -1);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  model_tmp_regst_ = nullptr;
  VirtualForwardCompActorInit(task_proto);
}

void ForwardCompActor::SwitchToHandlerInitModelTmpOrNormal() {
  if (model_tmp_regst_desc_id_ != -1) {
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerInitModelTmp);
  } else {
    SetMsgHandlerOfNormal();
  }
}

int ForwardCompActor::HandlerInitModel(const ActorMsg& msg) {
  Regst* model_regst = msg.regst();
  CHECK_EQ(model_regst->regst_desc_id(), model_regst_desc_id_);
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    exec_kernel.kernel->InitModelBlobs(
        GenDefaultKernelCtx(), parallel_ctx(),
        SnapshotMgr::Singleton()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          return model_regst->GetBlobByLbn(lbn);
        });
  }
  AsyncSendRegstMsgToProducer(model_regst);
  SwitchToHandlerInitModelTmpOrNormal();
  return 0;
}

int ForwardCompActor::HandlerInitModelTmp(const ActorMsg& msg) {
  Regst* model_tmp_regst = msg.regst();
  CHECK_EQ(model_tmp_regst->regst_desc_id(), model_tmp_regst_desc_id_);
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    exec_kernel.kernel->InitModelTmpBlobs(
        GenDefaultKernelCtx(), parallel_ctx(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          return model_tmp_regst->GetBlobByLbn(lbn);
        });
  }
  AsyncSendRegstMsgToProducer(model_tmp_regst);
  SetMsgHandlerOfNormal();
  return 0;
}

void ForwardCompActor::AsyncReturnAllReadableRegst() {
  CheckBeforeAsyncReturnAllReadableRegst();
  TryAsyncReturnModelRegst();
  TryAsyncReturnModelTmpRegst();
}

void ForwardCompActor::TryAsyncReturnModelTmpRegst() {
  if (model_tmp_regst_) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
}

}  // namespace oneflow
