#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_BUS_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_BUS_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ActorMsgBus final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgBus);
  ~ActorMsgBus() = default;

  void SendMsg(const ActorMsg& msg);
  void SendMsgWithoutCommNet(const ActorMsg& msg);

 private:
  friend class Global<ActorMsgBus>;
  ActorMsgBus() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_BUS_H_
