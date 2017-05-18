#include "thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  Join();
  id2actor_ptr_.clear();
}

void Thread::AddActor(const TaskProto& actor_proto) {
  TODO();
}

void Thread::PollMsgChannel() {
  ActorMsg msg;
  while (msg_channel_.Read(&msg) != -1) {
    id2actor_ptr_.at(msg.dst_actor_id())->ProcessMsg(msg);
  }
}

void Thread::Join() {
  thread_.join();
}

}  // namespace oneflow
