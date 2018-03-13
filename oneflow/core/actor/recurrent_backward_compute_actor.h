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
  void CheckBeforeAsyncReturnAllReadableRegst() override;
  void HandleTheRestOfRegstMsg(Regst*) override;

  Blob* HandleSpecialBnInOp(const std::string& bn_in_op) override;
  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;
  bool IsReadReady() override;
  void Act() override;

  bool RetFalseOrTerminate(Regst*) const;

  int64_t h0_regst_desc_id_;
  int64_t rec_in_regst_desc_id_;
  int64_t rec_out_diff_regst_desc_id_;

  std::queue<Regst*> h0_regsts_;
  Regst* rec_out_diff_regst_;
  HashMap<int64_t, std::deque<std::deque<Regst*>>> readable_deq_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECURRENT_BACKWARD_COMPUTE_ACTOR_H_
