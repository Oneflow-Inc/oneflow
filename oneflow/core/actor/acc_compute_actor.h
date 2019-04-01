#ifndef ONEFLOW_CORE_ACTOR_ACC_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACC_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

class AccCompActor final : public AccumulateCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompActor);
  AccCompActor() = default;
  ~AccCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACC_COMPUTE_ACTOR_H_
