#ifndef ONEFLOW_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
#define ONEFLOW_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_

#include "oneflow/actor/actor.h"

namespace oneflow {

class MdUpdtCompActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompActor);
  MdUpdtCompActor() = default;
  ~MdUpdtCompActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

private:

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
