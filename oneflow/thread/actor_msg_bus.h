#ifndef ONEFLOW_THREAD_ACTOR_MSG_BUS_H_
#define ONEFLOW_THREAD_ACTOR_MSG_BUS_H_

#include "common/blocking_channel.h"

namespace oneflow {

struct ActorMsg {
  uint64_t dst_actor_id;
  uint64_t register_id;
};

using Id2MsgChannelMap = 
      HashMap<uint64_t, std::unique_ptr<BlockingChannel<ActorMsg>>>;

class ActorMsgBus final {
public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgBus);
  ~ActorMsgBus() = default;

  static ActorMsgBus& Singleton() {
    static ActorMsgBus obj;
    return obj;
  }

  void InsertThrdLocIdMsgQPair(
      uint64_t thrd_loc_id, 
      std::unique_ptr<BlockingChannel<ActorMsg>> msg_queue);

  void SendMsg(const ActorMsg& msg);

private:
  void SendMsg(const ActorMsg& msg, uint64_t thrd_loc_id);

  ActorMsgBus() = default;
  Id2MsgChannelMap thrd_loc_id2msg_queue_;

};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_ACTOR_MSG_BUS_H_
