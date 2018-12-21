#ifndef ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

class ReduceConcatCompActor final : public NaiveActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatCompActor);
  ReduceConcatCompActor() = default;
  ~ReduceConcatCompActor() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_
