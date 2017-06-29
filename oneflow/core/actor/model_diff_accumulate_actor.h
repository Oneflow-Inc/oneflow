#ifndef ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class MdDiffAccActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccActor);
  MdDiffAccActor() = default;
  ~MdDiffAccActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

private:
  int HandleMdDiffAcc(const ActorMsg&);
  int HandleMdDiffAccWhenNoReadableRegstMsg(const ActorMsg&);

  void TryWardKernelAndSendMsg();

  ExecKernel clear_ek_;

};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
