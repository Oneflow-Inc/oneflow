#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class MdSaveCompActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

private:

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
