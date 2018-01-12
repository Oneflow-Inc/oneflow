#ifndef ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyHdActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  void InitDeviceCtx(const ThreadCtx&) override;

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return !pending_in_regst_.empty(); }
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  std::list<RegstEvent> CurActComsumedRegstEvents() const override;

  bool is_in_eord_;
  std::queue<Regst*> pending_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
