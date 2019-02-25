#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/actor/nccl_actor.h"
#include "oneflow/core/device/nccl_device_context.h"

namespace oneflow {

void NcclActor::VirtualActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    consumed_regst_desc_ids_.insert(consumed_regst_desc_ids_.end(),
                                    pair.second.regst_desc_id().cbegin(),
                                    pair.second.regst_desc_id().cend());
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    produced_regst_desc_ids_.push_back(pair.second.regst_desc_id());
  }
  OF_SET_MSG_HANDLER(&NcclActor::HandlerNormal);
}

void NcclActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
#ifdef WITH_CUDA
  CHECK_EQ(GetDeviceType(), DeviceType::kGPU);
  mut_device_ctx().reset(
      new NcclDeviceCtx(Global<NcclCommMgr>::Get()->NcclComm4ActorId(actor_id())));
#else
  UNIMPLEMENTED();
#endif  // WITH_CUDA
}

void NcclActor::Act() {
  NcclDeviceCtx* nccl_device_ctx = dynamic_cast<NcclDeviceCtx*>(mut_device_ctx().get());
  CHECK_NOTNULL(nccl_device_ctx);
  const KernelCtx kernel_ctx = GenDefaultKernelCtx();
  std::function<Regst*(int64_t)> Regst4RegstDescId =
      GetNaiveCurReadableWriteableRegst4RegstDescId();
  nccl_device_ctx->Enqueue(
      [this, kernel_ctx, Regst4RegstDescId]() { LaunchKernel(kernel_ctx, Regst4RegstDescId); });
}

void NcclActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t piece_id = GetPieceId4NaiveCurReadableDataRegst();
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
}

void NcclActor::LaunchKernel(const KernelCtx& kernel_ctx,
                             std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec()) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      CHECK(!(regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()));
      Regst* regst = Regst4RegstDescId(regst_desc_id_it->second);
      CHECK_NOTNULL(regst);
      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
      Blob* blob = regst->GetBlobByLbi(lbi);
      CHECK_NOTNULL(blob);
      return blob;
    });
  }
}

std::function<Regst*(int64_t)> NcclActor::GetNaiveCurReadableWriteableRegst4RegstDescId() {
  HashMap<int64_t, Regst*> regst_desc_id2regst;
  for (const int64_t regst_desc_id : consumed_regst_desc_ids_) {
    regst_desc_id2regst.emplace(regst_desc_id, GetNaiveCurReadable(regst_desc_id));
  }
  for (const int64_t regst_desc_id : produced_regst_desc_ids_) {
    regst_desc_id2regst.emplace(regst_desc_id, GetNaiveCurWriteable(regst_desc_id));
  }
  return [regst_desc_id2regst](int64_t regst_desc_id) -> Regst* {
    auto it = regst_desc_id2regst.find(regst_desc_id);
    if (it == regst_desc_id2regst.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  };
}

REGISTER_ACTOR(TaskType::kNcclAllReduce, NcclActor);
REGISTER_ACTOR(TaskType::kNcclReduceScatter, NcclActor);
REGISTER_ACTOR(TaskType::kNcclAllGather, NcclActor);

}  // namespace oneflow
