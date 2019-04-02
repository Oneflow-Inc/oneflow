#ifndef ONEFLOW_CORE_ACTOR_EVERY_NTH_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_EVERY_NTH_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class EveryNthCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EveryNthCompActor);
  EveryNthCompActor() = default;
  ~EveryNthCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override { return every_nth_; };

  int64_t every_nth_;
  int64_t current_nth_;
  int64_t cur_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_EVERY_NTH_COMPUTE_ACTOR_H_
