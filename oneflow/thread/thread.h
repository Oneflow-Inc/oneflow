#ifndef ONEFLOW_THREAD_THREAD_H_
#define ONEFLOW_THREAD_THREAD_H_

#include <memory>
#include <thread>
#include "common/util.h"
#include "common/channel.h"
#include "common/task.pb.h"
#include "actor/actor.h"
#include "actor/actor_msg_bus.h"

namespace oneflow {

class Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  virtual ~Thread();

  void AddActor(const TaskProto&);

  Channel<ActorMsg>* GetMsgChannelPtr() { return &msg_channel_; }

  void Join();

 protected:
  Thread() = default;
  std::thread& mut_thread() { return thread_; }
  void PollMsgChannel();

 private:

  std::thread thread_;
  Channel<ActorMsg> msg_channel_;
  HashMap<uint64_t, std::unique_ptr<Actor>> id2actor_ptr_;

};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_H_
