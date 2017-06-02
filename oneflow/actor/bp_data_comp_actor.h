#ifndef ONEFLOW_ACTOR_BP_DATA_COMP_ACTOR_H_
#define ONEFLOW_ACTOR_BP_DATA_COMP_ACTOR_H_

#include "actor/actor.h"

namespace oneflow {

class BpDataCompActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(BpDataCompActor);
  BpDataCompActor() = default;
  ~BpDataCompActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

private:

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_BP_DATA_COMP_ACTOR_H_
