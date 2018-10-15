#ifndef ONEFLOW_CORE_ACTOR_PACK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_PACK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class PackCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PackCompActor);
  PackCompActor() = default;
  ~PackCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  size_t total_pack_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_PACK_COMPUTE_ACTOR_H_
