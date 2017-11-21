#ifndef ONEFLOW_CORE_ACTOR_LOSS_RECORD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_LOSS_RECORD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class LossRecordActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordActor);
  LossRecordActor() = default;
  ~LossRecordActor() = default;

 private:
  void VirtualActorInit(const TaskProto&, const ThreadCtx&) override;

  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilReadAlwaysUnReady(const ActorMsg& msg) override {
    UNEXPECTED_RUN();
  }

  bool IsReadReady() override { return regst_ != nullptr; }
  void Act() override;

  Regst* regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_LOSS_RECORD_ACTOR_H_
