#ifndef ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardCompActor);
  BackwardCompActor() = default;
  ~BackwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilReadAlwaysUnReady(const ActorMsg&) override;

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override { TODO(); }
  void Act() override;

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  int64_t out_regst_desc_id_;
  HashMap<int64_t, std::queue<Regst*>> readable_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
