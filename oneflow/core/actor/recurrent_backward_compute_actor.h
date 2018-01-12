#ifndef ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_

#include <stack>
#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RecurrentBackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentBackwardCompActor);
  RecurrentBackwardCompActor() = default;
  ~RecurrentBackwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  void AsyncReturnModelRegstUntilMatchCurOutRegst();
  void TryUpdtInsertOrder(const Regst*);

  int64_t in_regst_desc_id_;
  std::deque<std::stack<Regst*>> in_regsts_;

  int64_t out_regst_desc_id_;
  std::deque<std::deque<Regst*>> out_regsts_;

  int64_t data_tmp_regst_desc_id_;
  std::deque<std::stack<Regst*>> data_tmp_regsts_;

  int64_t initial_hidden_regst_desc_id_;
  std::queue<Regst*> init_hid_regsts_;

  int64_t out_diff_regst_desc_id_;
  // regst in deque is ascending by col_id
  std::deque<std::deque<Regst*>> out_diff_regsts_;
  // 0:not set, 1:insert to back, -1:front
  int32_t insert_order_;

  int64_t rec_acc_diff_regst_desc_id_;  // recurrent accumulate diff
  Regst* rec_acc_diff_regst_;

  int64_t model_regst_desc_id_;
  std::queue<Regst*> model_regsts_;

  int64_t model_tmp_regst_desc_id_;
  Regst* model_tmp_regst_;

  bool is_out_diff_eord_ = false;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_
