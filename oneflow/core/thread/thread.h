#ifndef ONEFLOW_CORE_THREAD_THREAD_H_
#define ONEFLOW_CORE_THREAD_THREAD_H_

#include <memory>
#include <thread>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/actor/actor_message_bus.h"

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
  void PollMsgChannel(const ThreadContext& thread_ctx);

 private:

  std::thread thread_;
  Channel<ActorMsg> msg_channel_;
  HashMap<uint64_t, std::unique_ptr<Actor>> id2actor_ptr_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_THREAD_THREAD_H_
