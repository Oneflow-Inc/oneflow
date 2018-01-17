#ifndef ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

class LossAccCompActor final : public AccumulateCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccCompActor);
  LossAccCompActor() = default;
  ~LossAccCompActor() = default;

  void VirtualCompActorInit(const TaskProto& proto) override {
    AccumulateCompActor::Init(proto,
                              JobDesc::Singleton()->PieceNumOfPrintLoss(),
                              ColIdOrder::kAscending);
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_COMPUTE_ACTOR_H_
