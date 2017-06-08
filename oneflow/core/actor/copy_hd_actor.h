#ifndef ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyHdActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&, const ThreadContext&) override;

private:
  std::queue<std::shared_ptr<RegstWarpper>> waiting_in_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
