#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/actor/nccl_actor.h"
#include "oneflow/core/device/nccl_device_context.h"

namespace oneflow {

void NcclActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
#ifdef WITH_CUDA
  CHECK_EQ(GetDeviceType(), DeviceType::kGPU);
  // CHECK_EQ(GetLocalWorkStreamId(), 0);
  mut_device_ctx().reset(new NcclDeviceCtx(
      thread_ctx.g_cuda_stream.get(), Global<NcclCommMgr>::Get()->NcclComm4ActorId(actor_id())));
#else
  UNIMPLEMENTED();
#endif  // WITH_CUDA
}

REGISTER_ACTOR(TaskType::kNcclAllReduce, NcclActor);
REGISTER_ACTOR(TaskType::kNcclReduceScatter, NcclActor);
REGISTER_ACTOR(TaskType::kNcclAllGather, NcclActor);
REGISTER_ACTOR(TaskType::kNcclTupleBroadcast, NcclActor);
REGISTER_ACTOR(TaskType::kNcclTupleReduce, NcclActor);

}  // namespace oneflow
