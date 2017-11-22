#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

void MdUpdtCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  next_model_version_id_ = 0;
  // related_save_task_id_ = task_proto.related_save_task_id();
  random_seed_ = task_proto.random_seed();
  if (JobDesc::Singleton()->GetDeviceType() == DeviceType::kCPU) {
    mut_device_ctx().reset(new CpuDeviceCtx());
    MemcpyFunc =
        std::bind(&Memcpy<DeviceType::kCPU>, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3,
                  std::placeholders::_4, cudaMemcpyKind::cudaMemcpyHostToHost);
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
    MemcpyFunc = std::bind(&Memcpy<DeviceType::kGPU>, std::placeholders::_1,
                           std::placeholders::_2, std::placeholders::_3,
                           std::placeholders::_4,
                           cudaMemcpyKind::cudaMemcpyDeviceToDevice);
  }
  OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerBeforeInitializeModel);
}

int MdUpdtCompActor::HandlerBeforeInitializeModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kInitializeModel);
  HashSet<const Kernel*> kernels;
  auto CollectKernelsFromLbn = [&kernels](const std::string& lbn) {
    std::string op_name = GetOpNameFromLbn(lbn);
  };
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = reinterpret_cast<void*>(random_seed_);
  // model_regst
  if (model_regst_desc_id_ != -1) {
    Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
    model_regst->set_model_version_id(next_model_version_id_++);
    model_regst->ForEachLbn(CollectKernelsFromLbn);
    for (const Kernel* kernel : kernels) {
      kernel->InitModelBlobs(kernel_ctx, ParallelContext(),
                             SnapshotMgr::Singleton()->GetReadableSnapshot(),
                             [&](const std::string& bn_in_op) {
                               const std::string& lbn =
                                   kernel->Lbn4BnInOp(bn_in_op);
                               return model_regst->GetBlobPtrFromLbn(lbn);
                             });
    }
    if (JobDesc::Singleton()->IsTrain()) { AsyncCopyModelFromCurToNext(); }
  }
  kernels.clear();
  // model_tmp_regst
  Regst* model_tmp_regst = GetCurWriteableRegst(model_tmp_regst_desc_id_);
  if (model_tmp_regst) {
    model_tmp_regst->ForEachLbn(CollectKernelsFromLbn);
    for (const Kernel* kernel : kernels) {
      kernel->InitModelTmpBlobs(
          kernel_ctx, ParallelContext(), [&](const std::string& bn_in_op) {
            const std::string& lbn = kernel->Lbn4BnInOp(bn_in_op);
            return model_tmp_regst->GetBlobPtrFromLbn(lbn);
          });
    }
  }
  kernels.clear();
  // data_tmp_regst
  Regst* data_tmp_regst = GetCurWriteableRegst(data_tmp_regst_desc_id_);
  if (data_tmp_regst) {
    data_tmp_regst->ForEachLbn(CollectKernelsFromLbn);
    CHECK_EQ(kernels.size(), 1);
    auto mdupdt_kernel =
        static_cast<const ModelUpdtKernel*>(*(kernels.begin()));
    mdupdt_kernel->InitDataTmpBlobs(
        kernel_ctx, [&](const std::string& bn_in_op) {
          const std::string& lbn = mdupdt_kernel->Lbn4BnInOp(bn_in_op);
          return data_tmp_regst->GetBlobPtrFromLbn(lbn);
        });
  }
  //
  AsyncDo([]() { RuntimeCtx::Singleton()->mut_model_init_cnt().MinusOne(); });
  OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerBeforeSendInitialModel);
  return 0;
}

int MdUpdtCompActor::HandlerBeforeSendInitialModel(const ActorMsg& actor_msg) {
  CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kSendInitialModel);
  AsyncSendRegstMsgToConsumer();
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendEORDMsgToConsumers(model_tmp_regst_desc_id_);
  }
  if (JobDesc::Singleton()->IsTrain()) {
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerNormal);
  } else {
    AsyncSendEORDMsgToConsumers(model_regst_desc_id_);
    OF_SET_MSG_HANDLER(&MdUpdtCompActor::HandlerZombie);
  }
  return 0;
}

int MdUpdtCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    // CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = actor_msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      waiting_model_diff_acc_queue_.push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int MdUpdtCompActor::HandlerUntilReadAlwaysUnReady(const ActorMsg& actor_msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(actor_msg.regst()), 0);
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
  Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
  auto model_wpr = model_regst;
  model_regst->set_model_version_id(next_model_version_id_);
  Regst* data_tmp_regst = GetCurWriteableRegst(data_tmp_regst_desc_id_);
  auto data_tmp_wpr = data_tmp_regst;
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &next_model_version_id_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == model_regst_desc_id_) {
      return model_wpr;
    } else if (regst_desc_id == data_tmp_regst_desc_id_) {
      return data_tmp_wpr;
    } else {
      return model_diff_acc_wpr;
    }
  });
  AsyncSendRegstMsgToProducer(model_diff_acc_wpr);
  AsyncCopyModelFromCurToNext();
  if (next_model_version_id_ == JobDesc::Singleton()->TotalBatchNum()) {
    AsyncSendRegstMsgToConsumer(
        [this](int64_t actor_id) { return actor_id == related_save_task_id_; });
    CHECK(!IsReadReady());
    AsyncSendEORDMsgToConsumers(model_regst_desc_id_);
    TrySwitchToZombie();
  } else {
    if (next_model_version_id_ % JobDesc::Singleton()->NumOfBatchesInSnapshot()
        == 0) {
      AsyncSendRegstMsgToConsumer();
    } else {
      AsyncSendRegstMsgToConsumer([this](int64_t actor_id) {
        return actor_id != related_save_task_id_;
      });
    }
  }
  next_model_version_id_ += 1;
}

}  // namespace oneflow
