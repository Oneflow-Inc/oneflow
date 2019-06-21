#ifndef ONEFLOW_CORE_ACTOR_MODEL_INIT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_INIT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdInitCompActor : public CompActor {
 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override; 
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_INIT_COMPUTE_ACTOR_H_
