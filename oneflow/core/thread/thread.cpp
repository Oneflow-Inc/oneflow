#include "oneflow/core/thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  actor_thread_.join();
}

void Thread::AddActor(const TaskProto&) {
  TODO();
}

void Thread::PollMsgChannel(const ThreadContext& thread_ctx) {
  ActorMsg msg;
  while (msg_channel_.Receive(&msg) == 0) {
    auto actor_it = id2actor_ptr_.find(msg.dst_actor_id());
    int code = actor_it->second->ProcessMsg(msg, thread_ctx);
    if (code == 1) {
      // Be Careful: do not erase this actor
    }
    CHECK_EQ(code, 0);
  }
}

}  // namespace oneflow
