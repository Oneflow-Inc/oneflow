#ifndef ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_

#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

class NcclActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclActor);
  NcclActor() = default;
  ~NcclActor() override = default;

  void VirtualActorInit(const TaskProto&) override {
    OF_SET_MSG_HANDLER(&NcclActor::HandlerNormal);
  }

 private:
  void InitDeviceCtx(const ThreadCtx& thread_ctx) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void LaunchKernel(const KernelCtx& kernel_ctx, std::function<Regst*(int64_t)> Regst4RegstDescId);
  bool IsKernelLaunchSynchronized() override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_
