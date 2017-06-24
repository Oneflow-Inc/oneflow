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
  int HandleSaveModel(const ActorMsg&);

  int64_t model_regst_desc_id_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
