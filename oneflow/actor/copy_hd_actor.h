#ifndef ONEFLOW_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_ACTOR_COPY_HD_ACTOR_H_

#include "actor/actor.h"

namespace oneflow {

class CopyHdActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

private:

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_COPY_HD_ACTOR_H_
