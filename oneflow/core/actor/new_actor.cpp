#include "oneflow/core/actor/new_actor.h"

namespace oneflow {

namespace actor {

std::unique_ptr<NewActor> ConstructNewActor(const TaskProto& task_proto,
                                            const ThreadCtx& thread_ctx) {
  NewActor* rptr = NewObj<NewActor>(task_proto.task_type());
  rptr->Init(task_proto, thread_ctx);
  return std::unique_ptr<NewActor>(dynamic_cast<NewActor*>(rptr));
}

}  // namespace actor

}  // namespace oneflow
