#ifndef ONEFLOW_CORE_ACTOR_PIECE_SLICE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_PIECE_SLICE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class PieceSliceCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceCompActor);
  PieceSliceCompActor() = default;
  ~PieceSliceCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;

  size_t total_slice_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
  bool handle_instance_stack_bw_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_PIECE_SLICE_COMPUTE_ACTOR_H_
