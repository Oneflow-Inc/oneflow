#ifndef ONEFLOW_CORE_ACTOR_TICK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_TICK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class TickComputeActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TickComputeActor);
  TickComputeActor() = default;
  ~TickComputeActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override {}
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  int64_t piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_TICK_COMPUTE_ACTOR_H_
