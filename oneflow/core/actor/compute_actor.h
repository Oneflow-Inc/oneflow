#ifndef ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CompActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompActor);
  virtual ~CompActor() = default;

 protected:
  CompActor() = default;

  virtual void VirtualCompActorInit(const TaskProto& task_proto,
                                    const ThreadCtx& thread_ctx) {}

 private:
  void VirtualActorInit(const TaskProto& task_proto,
                        const ThreadCtx& thread_ctx) override {
    parallel_ctx_ = task_proto.parallel_ctx();
    VirtualCompActorInit(task_proto, thread_ctx);
  }

  ParallelContext parallel_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
