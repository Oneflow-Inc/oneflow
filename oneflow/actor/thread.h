#ifndef ONEFLOW_ACTOR_THREAD_H_
#define ONEFLOW_ACTOR_THREAD_H_

#include <memory>
#include <thread>
#include "common/util.h"
#include "common/channel.h"
#include "actor/task.pb.h"
#include "actor/actor.h"
#include "actor/actor_msg_bus.h"

namespace oneflow {

class Thread final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  Thread(): thread_([this]() { this->PollMsgChannel(); }) {}
  ~Thread();

  uint64_t thrd_loc_id() const { return thrd_loc_id_; }
  void set_thrd_loc_id(uint64_t thrd_loc_id) {
    thrd_loc_id_ = thrd_loc_id;
  }

  void AddActor(const TaskProto& actor_proto);

  Channel<ActorMsg>* GetMsgChannelPtr() { return &msg_channel_; }

  void Join();

 private:
  void PollMsgChannel();

  std::thread thread_;
  uint64_t thrd_loc_id_;
  Channel<ActorMsg> msg_channel_;
  HashMap<uint64_t, std::unique_ptr<Actor>> id2actor_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_THREAD_H_
