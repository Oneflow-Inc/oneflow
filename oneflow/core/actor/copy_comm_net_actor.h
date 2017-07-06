#ifndef ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleCopyCommNet(const ActorMsg&);
  int HandleCopyCommNetWhenNoReadableRegstMsg(const ActorMsg&);

  void TryWardKernelAndSendMsg();
  HashMap<int64_t, std::shared_ptr<RegstWarpper>> piece_id2waiting_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
