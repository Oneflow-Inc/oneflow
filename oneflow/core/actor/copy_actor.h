#ifndef ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyActor : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyActor);
  ~CopyActor() = default;

  virtual void Init(const TaskProto&, const ThreadCtx&) = 0;
  int ProcessMsg(const ActorMsg&) override;

protected:
  CopyActor() = default;

private:
  std::queue<std::shared_ptr<RegstWarpper>> waiting_in_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_
