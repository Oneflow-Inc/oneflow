#ifndef ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class DecodeRandomActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomActor);
  DecodeRandomActor() = default;
  ~DecodeRandomActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_
