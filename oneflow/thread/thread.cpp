#include "thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  Join();
}

void Thread::AddActor(const TaskProto& actor_proto) {
  TODO();
}

void Thread::PollMsgChannel() {
  ActorMsg msg;
  while (msg_channel_.Receive(&msg) != -1) {
    id2actor_ptr_.at(msg.dst_actor_id())->ProcessMsg(msg);
  }
}

void Thread::Join() {
  thread_.join();
}

}  // namespace oneflow
