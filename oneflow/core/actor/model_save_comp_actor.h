#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdSaveCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return model_regst_; }
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  int64_t model_regst_desc_id_;
  Regst* model_regst_;
  int64_t next_snapshot_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
