#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdSaveCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

  void VirtualCompActorInit(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilNoReadableRegst(const ActorMsg& msg) override {
    UNEXPECTED_RUN();
  }

  bool IsReadReady() override { return regst_ != nullptr; }
  void Act() override;

  int64_t model_regst_desc_id_;
  Regst* regst_;
  int64_t next_snapshot_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
