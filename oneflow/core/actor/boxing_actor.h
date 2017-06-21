#ifndef ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;
  int ProcessMsg(const ActorMsg&) override;

 private:
  int HandleInitDeviceCtx(const ActorMsg&);
  int HandleBoxing(const ActorMsg&);
  int HandleBoxingWhenNoReadableRegstMsg(const ActorMsg&);
  int HandleWaitUntilReadingCntEqualZero(const ActorMsg&);

  void TryWardKernelAndSendMsg();

  int (BoxingActor::*cur_msg_handle_)(const ActorMsg&);
  int num_of_subscribed_regsts_;
  int num_of_read_empty_;
  int num_of_eord_;
  // <regst_desc_id, queue<regst_wp>>
  HashMap<uint64_t, std::queue<std::shared_ptr<RegstWarpper>>> read_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
