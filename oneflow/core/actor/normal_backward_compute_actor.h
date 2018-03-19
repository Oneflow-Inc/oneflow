#ifndef ONEFLOW_CORE_ACTOR_NORMAL_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

class NormalBackwardCompActor final : public BackwardCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalBackwardCompActor);
  NormalBackwardCompActor() = default;
  ~NormalBackwardCompActor() = default;

 private:
  void VirtualBackwardCompActorInit(const TaskProto&) override;
  void CheckBeforeAsyncReturnAllReadableRegst() override;
  void HandleTheRestOfRegstMsg(Regst*) override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;
  bool IsReadReady() override;
  void Act() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
