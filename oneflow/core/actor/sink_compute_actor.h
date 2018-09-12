#ifndef ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class SinkCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkCompActor);
  SinkCompActor() = default;
  virtual ~SinkCompActor() = default;

 protected:
  virtual void VirtualSinkCompActorInit(const TaskProto&) {}
  virtual void* NewOther() { return nullptr; }
  virtual void DeleteOther(void*) {}

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_
