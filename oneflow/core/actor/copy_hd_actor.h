#ifndef ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyHdActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleNormal(const ActorMsg&) override;
  int HandleWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override { return !waiting_in_regst_.empty(); }
  void Act() override;

  std::queue<std::shared_ptr<RegstWrapper>> waiting_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
