#ifndef ONEFLOW_CORE_ACTOR_BOXING_COPY_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BOXING_COPY_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class BoxingCopyCompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyCompActor);
  BoxingCopyCompActor() = default;
  ~BoxingCopyCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BOXING_COPY_COMPUTE_ACTOR_H_
