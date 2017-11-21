#ifndef ONEFLOW_CORE_THREAD_THREAD_H_
#define ONEFLOW_CORE_THREAD_THREAD_H_

#include "oneflow/core/actor/actor.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

class Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  virtual ~Thread();

  void AddTask(const TaskProto&);

  Channel<ActorMsg>* GetMsgChannelPtr() { return &msg_channel_; }

  void JoinAllActor() { actor_thread_.join(); }

 protected:
  Thread() = default;
  std::thread& mut_actor_thread() { return actor_thread_; }
  void PollMsgChannel(const ThreadCtx& thread_ctx);
  void set_thrd_id(int64_t val) { thrd_id_ = val; }

 private:
  void ConstructActor(int64_t actor_id, const ThreadCtx& thread_ctx);

  HashMap<int64_t, TaskProto> id2task_;
  std::mutex id2task_mtx_;

  std::thread actor_thread_;
  Channel<ActorMsg> msg_channel_;
  HashMap<int64_t, std::unique_ptr<Actor>> id2actor_ptr_;

  int64_t thrd_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_H_
