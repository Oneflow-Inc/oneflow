#ifndef ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class UnpackCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnpackCompActor);
  UnpackCompActor() = default;
  ~UnpackCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;

  size_t total_unpack_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_
