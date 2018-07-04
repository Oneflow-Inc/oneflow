#ifndef ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

class MdSaveCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompActor);
  MdSaveCompActor() = default;
  ~MdSaveCompActor() = default;

 private:
  void VirtualSinkCompActorInit(const TaskProto&) override;
  void* NewOther() override;
  void DeleteOther(void*) override;

  int64_t next_snapshot_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_SAVE_COMPUTE_ACTOR_H_
