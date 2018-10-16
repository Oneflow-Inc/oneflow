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
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;
  int64_t ActNumForEachOutput(int64_t) const override { return total_pack_num_; }

  size_t total_pack_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
  bool handle_unpack_bw_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_PACK_COMPUTE_ACTOR_H_
