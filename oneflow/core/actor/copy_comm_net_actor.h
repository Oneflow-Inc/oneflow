#ifndef ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/copy_actor.h"

namespace oneflow {

class CopyCommNetActor final : public CopyActor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor() = default;

  int ProcessMsg(const ActorMsg&, const ThreadContext&) override;

private:

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
