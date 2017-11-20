#ifndef ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class ForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompActor);
  ForwardCompActor() = default;
  ~ForwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandlerInitModel();
  int HandlerInitModelTmp();
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  void AsyncSendMsgToModelAndModelTmpProducer();

  int64_t in_regst_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  Regst* model_regst_;
  Regst* model_tmp_regst_;
  std::queue<Regst*> in_;
  HashMap<int64_t, Regst*> readable_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
