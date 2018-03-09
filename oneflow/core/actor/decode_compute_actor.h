#ifndef ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class DecodeCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeCompActor);
  DecodeCompActor() = default;
  ~DecodeCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerWaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override { return !IsReadReady(); }
  void AsyncReturnAllReadableRegst() override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_
