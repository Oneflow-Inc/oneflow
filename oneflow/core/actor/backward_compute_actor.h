#ifndef ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BpDataCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BpDataCompActor);
  BpDataCompActor() = default;
  ~BpDataCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilReadAlwaysUnReady(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  void AsyncSendMsgToModelAndModelTmpProducer();

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t out_regst_desc_id_;
  // <regst_desc_id, queue<regst>>
  HashMap<int64_t, std::queue<Regst*>> read_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
