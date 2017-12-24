#ifndef ONEFLOW_CORE_ACTOR_BASIC_RNN_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BASIC_RNN_BACKWARD_COMPUTE_ACTOR_H_

#include <stack>
#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BasicRnnBackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnBackwardCompActor);
  BasicRnnBackwardCompActor() = default;
  ~BasicRnnBackwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  bool CheckModel_In_OutDiff_Activation(Regst*) const;
  void FillReadableWithIn_OutDiff_Model_Activation(Regst*);
  void UpdtModelStatusAfterAct();
  void RelModelByJudgingStatus(int64_t);  // Rel for Release

  int64_t in_regst_desc_id_;
  HashMap<int64_t, std::stack<Regst*>> pid2in_regsts_;

  int64_t out_regst_desc_id_;
  HashMap<int64_t, std::deque<Regst*>> pid2out_regsts_;

  int64_t initial_hidden_regst_desc_id_;
  HashMap<int64_t, Regst*> pid2init_hid_regsts_;

  int64_t out_diff_regst_desc_id_;
  // regst in deque is ascending by col_id
  HashMap<int64_t, std::deque<Regst*>> pid2out_diff_regsts_;
  bool is_insert_to_back_;

  int64_t rec_acc_diff_regst_desc_id_;  // recurrent accumulate diff
  HashMap<int64_t, Regst*> pid2rec_acc_diff_regsts_;

  int64_t model_regst_desc_id_;
  HashMap<int64_t, Regst*> model_vid2model_regst_;
  HashMap<int64_t, int64_t> model_vid2cnt_;
  // the only way to release a model regst is through model_vid2status_
  // except the last several unused model regsts
  // <model_vid, model_can_be_released>
  std::map<int64_t, bool> model_vid2status_;

  int64_t activation_regst_desc_id_;
  HashMap<int64_t, std::stack<Regst*>> pid2activation_regsts_;

  bool is_out_diff_eord_ = false;
  HashMap<int64_t, Regst*> readable_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BASIC_RNN_BACKWARD_COMPUTE_ACTOR_H_
