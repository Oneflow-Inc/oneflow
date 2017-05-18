#ifndef ONEFLOW_THREAD_ACTOR_MSG_BUS_H_
#define ONEFLOW_THREAD_ACTOR_MSG_BUS_H_

#include <stdint.h>
#include "common/util.h"
#include "actor/actor_message.pb.h"

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

#endif  // ONEFLOW_THREAD_ACTOR_MSG_BUS_H_
