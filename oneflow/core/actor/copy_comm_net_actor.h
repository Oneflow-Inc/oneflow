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
  int ProcessMsg(const ActorMsg&) override;

private:
  int HandleCopyCommNet(const ActorMsg&);
  int HandleCopyCommNetWhenNoReadableRegstMsg(const ActorMsg&);
  int HandleWaitUntilReadingCntEqualZero(const ActorMsg&);

  void TryWardKernelAndSendMsg();
  int (CopyCommNetActor::*cur_msg_handle_)(const ActorMsg&);
  HashMap<uint64_t, std::shared_ptr<RegstWarpper>> id2waiting_in_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
