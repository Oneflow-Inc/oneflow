#ifndef ONEFLOW_CORE_ACTOR_ACC_TICK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACC_TICK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class AccTickCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccTickCompActor);
  AccTickCompActor() = default;
  virtual ~AccTickCompActor() = default;

 protected:
  void VirtualCompActorInit(const TaskProto& proto) override;
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  int32_t acc_cnt_;
  int32_t max_acc_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACC_TICK_COMPUTE_ACTOR_H_
