#ifndef ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class GatherBackwardActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherBackwardActor);
  GatherBackwardActor() = default;
  ~GatherBackwardActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return !pending_in_regst_.empty(); };
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  bool is_in_eord_;
  std::queue<Regst*> pending_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_
