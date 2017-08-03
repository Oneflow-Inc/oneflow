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
  related_save_task_id_ = task_proto.related_save_task_id();
  set_num_of_remaining_eord(1);
  mut_num_of_read_empty() = 1;
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
    MemcpyFunc = std::bind(&KernelUtil<DeviceType::kCPU, float>::Memcpy,
                           std::placeholders::_1, std::placeholders::_2,
                           std::placeholders::_3, std::placeholders::_4,
                           cudaMemcpyKind::cudaMemcpyHostToHost);
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
    MemcpyFunc = std::bind(&KernelUtil<DeviceType::kGPU, float>::Memcpy,
                           std::placeholders::_1, std::placeholders::_2,
                           std::placeholders::_3, std::placeholders::_4,
                           cudaMemcpyKind::cudaMemcpyDeviceToDevice);
  }
  OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerBeforeInitializeModel);
}

int MdUpdtCompActor::HandlerBeforeInitializeModel(const ActorMsg& actor_msg) {
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
  if (model_tmp_regst) { model_tmp_regst->ForEachLbn(CollectKernelsFromLbn); }

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
  if (JobDesc::Singleton()->is_train()) { AsyncCopyModelFromCurToNext(); }
  AsyncDo([]() { RuntimeCtx::Singleton()->mut_model_init_cnt().MinusOne(); });
  OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerBeforeSendInitialModel);
  return 0;
}

int MdUpdtCompActor::HandlerBeforeSendInitialModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  AsyncSendReadableRegstMsg();
  if (model_tmp_regst_desc_id_ != -1) {
    SetReadOnlyForRegstDescId(model_tmp_regst_desc_id_);
    AsyncSendEORDMsgToConsumers(model_tmp_regst_desc_id_);
  }
  if (JobDesc::Singleton()->is_train()) {
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerNormal);
  } else {
    AsyncSendEORDMsgToConsumers(model_regst_desc_id_);
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerZombie);
  }
  return 0;
}

int MdUpdtCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_wp = actor_msg.regst_wrapper();
    if (TryUpdtStateAsProducedRegst(regst_wp->regst_raw_ptr()) != 0) {
      waiting_model_diff_acc_queue_.push(regst_wp);
      mut_num_of_read_empty() = 0;
      VLOG(4) << "model update actor " << actor_id() << " "
              << "receive readable regst " << regst_wp->regst_raw_ptr() << ", "
              << "regst_desc_id:" << regst_wp->regst_desc_id() << ", "
              << "current num_of_read_empty:" << num_of_read_empty();
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int MdUpdtCompActor::HandlerWaitUntilNoReadableRegst(
    const ActorMsg& actor_msg) {
  CHECK_EQ(
      TryUpdtStateAsProducedRegst(actor_msg.regst_wrapper()->regst_raw_ptr()),
      0);
  ActUntilFail();
  if (waiting_model_diff_acc_queue_.empty()) {
    AsyncSendEORDMsgToConsumers(model_regst_desc_id_);
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerZombie);
  }
  return 0;
}

void MdUpdtCompActor::Act() {
  auto model_diff_acc_wpr = waiting_model_diff_acc_queue_.front();
  waiting_model_diff_acc_queue_.pop();
  mut_num_of_read_empty() = waiting_model_diff_acc_queue_.empty();
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  auto model_wpr = std::make_shared<LocalRegstWrapper>(model_regst);
  model_regst->set_model_version_id(next_model_version_id_);
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [&](int64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        if (regst_desc_id == model_regst_desc_id_) {
          return model_wpr;
        } else {
          return model_diff_acc_wpr;
        }
      });
  AsyncSendRegstMsgToProducer(model_diff_acc_wpr);
  AsyncCopyModelFromCurToNext();
  if (next_model_version_id_ == JobDesc::Singleton()->total_batch_num()) {
    AsyncSendReadableRegstMsg(
        [this](int64_t actor_id) { return actor_id == related_save_task_id_; });
    CHECK(!IsReadReady());
    AsyncSendEORDMsgToConsumers(model_regst_desc_id_);
    TrySwitchToZombie();
  } else {
    if (next_model_version_id_
            % JobDesc::Singleton()->num_of_batches_in_snapshot()
        == 0) {
      AsyncSendReadableRegstMsg();
    } else {
      AsyncSendReadableRegstMsg([this](int64_t actor_id) {
        return actor_id != related_save_task_id_;
      });
    }
  }
  next_model_version_id_ += 1;
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
