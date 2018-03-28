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
  int HandlerInitModelAndModelTmp(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void InitOtherModel();

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void Act() override;

  void UpdateModelRegstPtr(Regst* regst);

  void AsyncReturnModelRegst();
  void TryAsyncReturnModelRegst();
  void TryAsyncReturnModelTmpRegst();
  void TrySendMsgToOtherModelSaveActor(const int64_t piece_id);

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  bool is_in_eord_;
  int64_t in_regst_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t other_model_regst_desc_id_;
  int64_t random_seed_;
  Regst* model_regst_;
  Regst* model_tmp_regst_;
  std::queue<Regst*> pending_in_regsts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FORWARD_COMPUTE_ACTOR_H_
