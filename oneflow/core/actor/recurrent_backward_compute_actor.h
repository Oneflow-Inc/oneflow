#ifndef ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_

#include <stack>
#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

class RecurrentBackwardCompActor final : public BackwardCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentBackwardCompActor);
  RecurrentBackwardCompActor() = default;
  ~RecurrentBackwardCompActor() = default;

 private:
  void VirtualBackwardCompActorInit(const TaskProto&) override;

  int HandlerNormal(const ActorMsg&) override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  void TryUpdtColIdOrder(const Regst*);

  int64_t h0_regst_desc_id_;
  int64_t rec_out_diff_regst_desc_id_;

  std::queue<Regst*> h0_regsts_;
  std::deque<std::stack<Regst*>> in_regsts_;
  std::deque<std::deque<Regst*>> out_regsts_;
  std::deque<std::stack<Regst*>> data_tmp_regsts_;
  // regst in deque is ascending by col_id
  std::deque<std::deque<Regst*>> out_diff_regsts_;
  std::queue<Regst*> model_regsts_;
  Regst* rec_out_diff_regst_;
  Regst* model_tmp_regst_;

  ColIdOrder order_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_
