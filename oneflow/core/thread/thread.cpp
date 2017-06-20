#include "oneflow/core/thread/thread.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

Thread::~Thread() {
  actor_thread_.join();
}

void Thread::AddTask(const TaskProto& task) {
  CHECK(id2task_.emplace(task.id(), task).second);
}

void Thread::PollMsgChannel(const ThreadCtx& thread_ctx) {
  ActorMsg msg;
  while (msg_channel_.Receive(&msg) == 0) {
    uint64_t actor_id = msg.dst_actor_id();
    auto actor_it = id2actor_ptr_.find(actor_id);
    if (actor_it == id2actor_ptr_.end()) {
      uint64_t task_id = IDMgr::Singleton().TaskId4ActorId(actor_id);
      auto task_it = id2task_.find(task_id);
      auto emplace_ret = id2actor_ptr_.emplace(
          actor_id, ConstructActor(task_it->second, thread_ctx));
      actor_it = emplace_ret.first;
      CHECK(emplace_ret.second);
      id2task_.erase(task_it);
    }
    int process_msg_ret = actor_it->second->ProcessMsg(msg);
    if (process_msg_ret == 1) {
      // Be Careful: do not erase this actor
    }
    CHECK_EQ(process_msg_ret, 0);
  }
  id2actor_ptr_.clear();
}

}  // namespace oneflow
