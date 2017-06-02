#include "oneflow/thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  Join();
}

void Thread::AddActor(const TaskProto&) {
  TODO();
}

void Thread::Join() {
  thread_.join();
}

void Thread::PollMsgChannel() {
  ActorMsg msg;
  while (msg_channel_.Receive(&msg) == 0) {
    id2actor_ptr_.at(msg.dst_actor_id())->ProcessMsg(msg);
  }
}

}  // namespace oneflow
