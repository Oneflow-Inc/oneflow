#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MSG_BUS_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MSG_BUS_H_

#include <stdint.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

class ActorMsgBus final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgBus);
  ~ActorMsgBus() = default;

  static ActorMsgBus& Singleton() {
    static ActorMsgBus obj;
    return obj;
  }

  void SendMsg(const ActorMsg& msg);

 private:
  ActorMsgBus() = default;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_ACTOR_MSG_BUS_H_
