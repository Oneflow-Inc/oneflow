#ifndef ONEFLOW_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

private:

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_COPY_COMM_NET_ACTOR_H_
