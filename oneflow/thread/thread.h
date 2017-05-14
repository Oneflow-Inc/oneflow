#ifndef ONEFLOW_THREAD_THREAD_H_
#define ONEFLOW_THREAD_THREAD_H_

#include "common/util.h"
#include "common/blocking_channel.h"
#include "task/task.pb.h"
#include "thread/actor_msg_bus.h"

namespace oneflow {

// TODO(liuguo): use real Actor class
class Actor {
  public:
    Actor(const TaskProto& proto) {}
    uint64_t actor_id() { return 0; }
};

class Thread {
public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  Thread() = default;
  virtual ~Thread() = {
    msg_queue_.Close();
    id2actor_ptr_.clear();
  }

  uint64_t thrd_loc_id() const { return thrd_loc_id_; }
  void set_thrd_loc_id(uint64_t thrd_loc_id) {
    thrd_loc_id_ = thrd_loc_id;
  }

  void AddActor(const TaskProto& actor_proto);
  Actor* GetActorPtr4Id(uint64_t actor_id) {
    return id2actor_ptr_.at(actor_id).get();
  }

  BlockingChannel<ActorMsg>& GetMsgQueue() { return msg_queue_; }

  void join();

private:
  uint64_t thrd_loc_id_;
  BlockingChannel<ActorMsg> msg_queue_;
  HashMap<uint64_t, std::unique_ptr<Actor>> id2actor_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_H_
