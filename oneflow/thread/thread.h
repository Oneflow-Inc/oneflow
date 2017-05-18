#ifndef ONEFLOW_THREAD_THREAD_H_
#define ONEFLOW_THREAD_THREAD_H_

#include <thread>
#include "common/util.h"
#include "common/blocking_channel.h"
#include "actor/task.pb.h"
#include "actor/actor.h"
#include "thread/actor_msg_bus.h"

namespace oneflow {

class Thread {
public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  Thread(): thread_([this]() {this->PollMsgChannel(); }) {};
  virtual ~Thread();

  uint64_t thrd_loc_id() const { return thrd_loc_id_; }
  void set_thrd_loc_id(uint64_t thrd_loc_id) {
    thrd_loc_id_ = thrd_loc_id;
  }

  void AddActor(const TaskProto& actor_proto);
  Actor* GetActorPtr4Id(uint64_t actor_id) {
    return id2actor_ptr_.at(actor_id).get();
  }

  BlockingChannel<ActorMsg>* GetMsgChannelPtr() { return &msg_channel_; }

  void Join();

private:
  void PollMsgChannel();

  std::thread thread_;
  uint64_t thrd_loc_id_;
  BlockingChannel<ActorMsg> msg_channel_;
  HashMap<uint64_t, std::unique_ptr<Actor>> id2actor_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_H_
