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

  virtual void Init(const TaskProto& task_proto,
                    const ThreadCtx& thread_ctx) override {
    Actor::Init(task_proto, thread_ctx);
    parallel_id_ = task_proto.parallel_id();
  }

  ParallelPolicy parallel_policy() const { return parallel_policy_; }
  int64_t parallel_id() const { return parallel_id_; }
  int64_t parallel_num() const { return parallel_num_; }

 private:
  ParallelPolicy parallel_policy_;
  int64_t parallel_id_;
  int64_t parallel_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
