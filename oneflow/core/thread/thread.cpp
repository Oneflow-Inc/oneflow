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
    id2actor_ptr_.at(msg.dst_actor_id())->ProcessMsg(msg, thread_ctx);
  }
}

}  // namespace oneflow
