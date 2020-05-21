#ifndef ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CompActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompActor);
  virtual ~CompActor() = default;

 protected:
  CompActor() = default;

  virtual void VirtualCompActorInit(const TaskProto& task_proto) {}

 private:
  void VirtualActorInit(const TaskProto& task_proto) override { VirtualCompActorInit(task_proto); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
