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
  int HandlerNormal(const ActorMsg&) override;

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  HashMap<int64_t, std::queue<Regst*>> readable_regsts_;
  int64_t readable_regst_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
