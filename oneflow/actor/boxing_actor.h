#ifndef ONEFLOW_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_ACTOR_BOXING_ACTOR_H_

#include "actor/actor.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

 private:

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_BOXING_ACTOR_H_
