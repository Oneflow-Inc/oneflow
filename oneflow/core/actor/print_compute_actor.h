#ifndef ONEFLOW_CORE_ACTOR_PRINT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_PRINT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

class PrintCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintCompActor);
  PrintCompActor() = default;
  ~PrintCompActor() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_PRINT_COMPUTE_ACTOR_H_
