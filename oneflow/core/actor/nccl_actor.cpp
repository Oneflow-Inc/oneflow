#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/device/nccl_device_context.h"
#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclActor final : public NaiveActor {
 public:
  NcclActor() = default;
  ~NcclActor() override = default;

 private:
  void InitDeviceCtx(const ThreadCtx& thread_ctx) override;
};

void NcclActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  CHECK_EQ(GetDeviceType(), DeviceType::kGPU);
  mut_device_ctx().reset(
      new NcclDeviceCtx(Global<NcclCommMgr>::Get()->NcclComm4ActorId(actor_id())));
}

REGISTER_ACTOR(TaskType::kNcclAllReduce, NcclActor);
REGISTER_ACTOR(TaskType::kNcclReduceScatter, NcclActor);
REGISTER_ACTOR(TaskType::kNcclAllGather, NcclActor);

REGISTER_ACTOR(TaskType::kNcclTupleBroadcast, NcclActor);
REGISTER_ACTOR(TaskType::kNcclTupleReduce, NcclActor);

REGISTER_ACTOR(TaskType::kNcclBoxingAllReduce, NcclActor);
REGISTER_ACTOR(TaskType::kNcclBoxingReduceScatter, NcclActor);
REGISTER_ACTOR(TaskType::kNcclBoxingAllGather, NcclActor);

#endif  // WITH_CUDA

}  // namespace oneflow
