#ifndef ONEFLOW_CORE_ACTOR_INSTANCE_STACK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_INSTANCE_STACK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class InstanceStackCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InstanceStackCompActor);
  InstanceStackCompActor() = default;
  ~InstanceStackCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;
  int64_t ActNumForEachOutput(int64_t) const override { return total_stack_num_; }

  size_t total_stack_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
  bool handle_piece_slice_bw_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_INSTANCE_STACK_COMPUTE_ACTOR_H_
