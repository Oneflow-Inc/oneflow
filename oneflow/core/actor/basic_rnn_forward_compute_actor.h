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

  void UpdateModelRegstPtr(Regst* regst);

  void AsyncReturnModelRegst();
  void TryAsyncReturnModelRegst();
  void TryAsyncReturnModelTmpRegst();

  void UpdtInAndModelStates();

  bool is_in_eord_;

  int64_t in_regst_desc_id_;
  std::map<int64_t, std::queue<Regst*>> pid2in_regsts_;  // <piece_id, in_regst>
  // must be increasing in iteration, so not using HashMap

  int64_t initial_hidden_regst_desc_id_;
  std::queue<Regst*> initial_hidden_regsts_;

  int64_t model_regst_desc_id_;
  Regst* latest_model_regst_;
  HashMap<int64_t, Regst*> pid2model_regst_;
  HashMap<Regst*, int64_t> model_regst2cnt_;
  std::set<Regst*> models_to_be_released_;

  int64_t out_regst_desc_id_;
  HashMap<int64_t, Regst*> pid2out_regst_;

  HashMap<int64_t, Regst*> readable_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BASIC_RNN_FORWARD_COMPUTE_ACTOR_H_
