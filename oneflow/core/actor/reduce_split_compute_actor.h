#ifndef ONEFLOW_CORE_ACTOR_REDUCE_SPLIT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_SPLIT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

class ReduceSplitCompActor final : public NaiveActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitCompActor);
  ReduceSplitCompActor() = default;
  ~ReduceSplitCompActor() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_SPLIT_COMPUTE_ACTOR_H_
