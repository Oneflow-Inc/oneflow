#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/actor/nccl_actor.h"
#include "oneflow/core/device/nccl_device_context.h"

namespace oneflow {

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
  const int64_t in_regst_desc_id = Name2SoleRegstDescId("in");
  const int64_t out_regst_desc_id = Name2SoleRegstDescId("out");
  Regst* in_regst = GetNaiveCurReadable(in_regst_desc_id);
  Regst* out_regst = GetNaiveCurWriteable(out_regst_desc_id);
  std::function<Regst*(int64_t)> Regst4RegstDescId = [=](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == in_regst_desc_id) {
      return in_regst;
    } else if (regst_desc_id == out_regst_desc_id) {
      return out_regst;
    } else {
      return nullptr;
    }
  };
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

REGISTER_ACTOR(TaskType::kNcclAllReduce, NcclActor);
REGISTER_ACTOR(TaskType::kNcclReduceScatter, NcclActor);
REGISTER_ACTOR(TaskType::kNcclAllGather, NcclActor);

}  // namespace oneflow
