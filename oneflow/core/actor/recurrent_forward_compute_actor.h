#ifndef ONEFLOW_CORE_ACTOR_RECURRENT_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECURRENT_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

class RecurrentForwardCompActor final : public ForwardCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentForwardCompActor);
  RecurrentForwardCompActor() = default;
  ~RecurrentForwardCompActor() = default;

 private:
  int HandlerNormal(const ActorMsg&) override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void Act() override;

  void VirtualForwardCompActorInit(const TaskProto&) override;
  void TryAsyncReturnModelRegst() override;
  void CheckBeforeAsyncReturnAllReadableRegst() override;

  int64_t rec_in_regst_desc_id_;
  int64_t h0_regst_desc_id_;

  std::queue<Regst*> in_regsts_;
  std::queue<Regst*> h0_regsts_;
  Regst* latest_model_regst_;
  Regst* cur_model_regst_;
  Regst* rec_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECURRENT_FORWARD_COMPUTE_ACTOR_H_
