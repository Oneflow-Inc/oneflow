#include "thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  Join();
  for (auto& pair : id2actor_ptr_) {
    pair.second = nullptr;
  }
  id2actor_ptr_.clear();
}

void Thread::AddActor(const TaskProto& actor_proto) {
  TODO();
}

void Thread::ProcessMsgQueue() {
  ActorMsg msg;
  while (msg_queue_.Read(&msg) != -1) {
    id2actor_ptr_.at(msg.dst_actor_id())->ProcessMsg(msg);
  }
}

void Thread::Join() {
  thread_.join();
}

}  // namespace oneflow
