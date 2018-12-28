#ifndef ONEFLOW_CORE_ACTOR_LOSS_PRINT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_LOSS_PRINT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

class LossPrintCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintCompActor);
  LossPrintCompActor() = default;
  ~LossPrintCompActor() override = default;

 private:
  void VirtualSinkCompActorInit(const TaskProto&) override { timestamp_ = 0; }
  void* NewOther() override { return &timestamp_; }

  double timestamp_ = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_LOSS_PRINT_COMPUTE_ACTOR_H_
