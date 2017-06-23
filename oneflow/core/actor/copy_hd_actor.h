#ifndef ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyHdActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdActor);
  CopyHdActor() = default;
  ~CopyHdActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;
  int ProcessMsg(const ActorMsg&) override;

private:
  int HandleCopyHd(const ActorMsg&);
  int HandleCopyHdWhenNoReadableRegstMsg(const ActorMsg&);
  int HandleWaitUntilReadingCntEqualZero(const ActorMsg&);

  void TryWardKernelAndSendMsg();
  int (CopyHdActor::*cur_msg_handle_)(const ActorMsg&);
  std::queue<std::shared_ptr<RegstWarpper>> waiting_in_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
