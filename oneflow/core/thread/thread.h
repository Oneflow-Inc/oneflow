#ifndef ONEFLOW_CORE_THREAD_THREAD_H_
#define ONEFLOW_CORE_THREAD_THREAD_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/thread/thread_context.h"
#include "oneflow/core/actor/actor.h"

namespace oneflow {

namespace extension {
class ThreadExtensionContext;
}

class Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Thread);
  virtual ~Thread();

  void AddTask(const TaskProto&);

  Channel<ActorMsg>* GetMsgChannelPtr() { return &msg_channel_; }
  void EnqueueActorMsg(const ActorMsg& msg);

  void JoinAllActor() { actor_thread_.join(); }

  void set_thread_ext_ctx(std::shared_ptr<extension::ThreadExtensionContext> new_ctx) {
    thread_ext_ctx_ = new_ctx;
  }
  const std::shared_ptr<extension::ThreadExtensionContext> get_thread_ext_ctx() const {
    return thread_ext_ctx_;
  }

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
  std::queue<ActorMsg> local_msg_queue_;

  int64_t thrd_id_;
  std::shared_ptr<extension::ThreadExtensionContext> thread_ext_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_H_
