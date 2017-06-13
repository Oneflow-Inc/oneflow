#ifndef ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyActor : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyActor);
  ~CopyActor() = default;

  void Init(const TaskProto&) override;
  virtual void ProcessMsg(const ActorMsg&, const ThreadContext&) = 0;

protected:
  CopyActor() = default;
  void ProcessMsgWithKernelCtx(const ActorMsg& msg, const KernelCtx& kernel_ctx);

private:
  std::queue<std::shared_ptr<RegstWarpper>> waiting_in_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_
