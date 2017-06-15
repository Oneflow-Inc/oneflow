#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdSaveCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

  void Init(const TaskProto&) override;
  int ProcessMsg(const ActorMsg&, const ThreadContext&) override;

 private:
   int HandleBeforeInitDeviceCtx(const ActorMsg&, const ThreadContext&);
   int HandleSaveModel(const ActorMsg&, const ThreadContext&);
   int (MdSaveCompActor::*cur_msg_handle_)(const ActorMsg&, const ThreadContext&);

  uint64_t model_regst_desc_id_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMP_ACTOR_H_
