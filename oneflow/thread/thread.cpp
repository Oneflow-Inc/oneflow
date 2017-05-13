#include "thread/thread.h"

namespace oneflow {

void Thread::AddActor(const TaskProto& actor_proto) {
  auto actor = of_make_unique<Actor>(actor_proto);
  CHECK(id2actor_ptr_.emplace(actor->actor_id(), std::move(actor)).second);
}

}  // namespace oneflow
