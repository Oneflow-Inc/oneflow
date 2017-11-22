#ifndef ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class ForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompActor);
  ForwardCompActor() = default;
  ~ForwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  void SwitchToHandlerInitModelTmpOrNormal();
  int HandlerInitModel(const ActorMsg&);
  int HandlerInitModelTmp(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilReadAlwaysUnReady(const ActorMsg&) override;

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void Act() override;

  void UpdateModelRegstPtr(Regst* regst);

  void AsyncReturnModelRegst();
  void TryAsyncReturnModelRegst();
  void TryAsyncReturnModelTmpRegst();

  int64_t in_regst_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  Regst* model_regst_;
  Regst* model_tmp_regst_;
  std::queue<Regst*> pending_in_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_
