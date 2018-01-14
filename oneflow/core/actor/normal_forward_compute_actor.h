#ifndef ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

class NormalForwardCompActor final : public ForwardCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalForwardCompActor);
  NormalForwardCompActor() = default;
  ~NormalForwardCompActor() = default;

 private:
  int HandlerNormal(const ActorMsg&) override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void Act() override;

  void VirtualForwardCompActorInit(const TaskProto&) override;
  void SetMsgHandlerOfNormal() override;
  void TryAsyncReturnModelRegst() override;
  void CheckBeforeAsyncReturnAllReadableRegst() override;

  void UpdateModelRegstPtr(Regst* regst);
  void AsyncReturnModelRegst();

  Regst* model_regst_;
  std::queue<Regst*> pending_in_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_
