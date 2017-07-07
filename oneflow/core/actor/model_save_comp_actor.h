#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdSaveCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleNormal(const ActorMsg&) override;
  int HandleWaitUntilNoReadableRegst(const ActorMsg& msg) override {
    UNEXPECTED_RUN();
  }

  bool IsReadReady() override { return regst_wrapper_ != nullptr; }
  void Act() override;
  int ProcessEord() override { TODO(); }

  int64_t model_regst_desc_id_;
  std::shared_ptr<RegstWrapper> regst_wrapper_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
