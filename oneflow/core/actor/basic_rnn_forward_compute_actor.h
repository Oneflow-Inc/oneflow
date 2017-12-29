#ifndef ONEFLOW_CORE_ACTOR_BASIC_RNN_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BASIC_RNN_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BasicRnnForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnForwardCompActor);
  BasicRnnForwardCompActor() = default;
  ~BasicRnnForwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  void SwitchToHandlerInitModelTmpOrNormal();
  int HandlerInitModel(const ActorMsg&);
  int HandlerInitModelTmp(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  void UpdtInAndModelStates();

  bool is_in_eord_;

  int64_t in_regst_desc_id_;
  std::queue<Regst*> in_regsts_;

  int64_t initial_hidden_regst_desc_id_;
  std::queue<Regst*> initial_hidden_regsts_;

  int64_t model_regst_desc_id_;
  Regst* latest_model_regst_;
  Regst* cur_model_regst_;

  int64_t out_regst_desc_id_;
  Regst* out_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BASIC_RNN_FORWARD_COMPUTE_ACTOR_H_
