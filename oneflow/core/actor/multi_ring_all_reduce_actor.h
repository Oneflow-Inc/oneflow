#ifndef ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class MultiRingAllReduceActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiRingAllReduceActor);
  MultiRingAllReduceActor() = default;
  ~MultiRingAllReduceActor() override = default;

 protected:
  void VirtualActorInit(const TaskProto&) override;
  int HandlerAllReduce(const ActorMsg& msg);

  int64_t num_rings_ = -1;
  int64_t num_steps_ = -1;
  int64_t current_ring_id_ = -1;
  int64_t current_step_id_ = -1;

  bool in_regst_eord_ = false;
  int64_t in_regst_desc_id_ = -1;
  int64_t out_regst_desc_id_ = -1;
  int64_t out_regst_desc_reading_cnt_ = -1;
  std::deque<Regst*> in_regst_deque_;
  Regst* out_regst_ = nullptr;
  std::vector<bool> send_regst_ready_;
  std::vector<bool> recv_regst_ready_;
  std::vector<Regst*> send_regst_;
  std::vector<Regst*> recv_regst_;
  HashMap<int64_t, std::pair<bool, int64_t>> regst_desc_id2send_or_recv7ring_id_;

  MultiRingAllReduceKernelConf multi_ring_all_reduce_kernel_conf_;
  KernelConf kernel_conf_;
  std::vector<int64_t> send_regst_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_
