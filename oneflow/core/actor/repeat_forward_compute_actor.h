#ifndef ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RepeatForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatForwardCompActor);
  RepeatForwardCompActor() = default;
  ~RepeatForwardCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const override;

  int64_t repeat_num_;
  int64_t repeat_count_;
  int64_t cur_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REPEAT_FORWARD_COMPUTE_ACTOR_H_
