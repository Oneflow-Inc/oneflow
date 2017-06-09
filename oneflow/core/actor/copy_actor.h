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
  void ProcessMsgAndWardKernel(const ActorMsg& msg, const KernelContext& kernel_ctx);

private:
  std::queue<std::shared_ptr<RegstWarpper>> waiting_in_regst_;
  uint64_t waiting_piece_id_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_ACTOR_H_
