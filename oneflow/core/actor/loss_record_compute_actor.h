#ifndef ONEFLOW_CORE_ACTOR_LOSS_RECORD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_LOSS_RECORD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class LossRecordCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordCompActor);
  LossRecordCompActor() = default;
  ~LossRecordCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return loss_acc_regst_; }
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  Regst* loss_acc_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_LOSS_RECORD_COMPUTE_ACTOR_H_
