#ifndef ONEFLOW_CORE_ACTOR_REPEAT_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REPEAT_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RepeatBackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatBackwardCompActor);
  RepeatBackwardCompActor() = default;
  ~RepeatBackwardCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override { return repeat_num_; };
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  int64_t repeat_num_;
  int64_t acc_count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REPEAT_BACKWARD_COMPUTE_ACTOR_H_
