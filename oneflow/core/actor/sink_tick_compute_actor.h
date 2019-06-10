#ifndef ONEFLOW_CORE_ACTOR_SINK_TICK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SINK_TICK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

class SinkTickCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkTickCompActor);
  SinkTickCompActor() = default;
  ~SinkTickCompActor() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SINK_TICK_COMPUTE_ACTOR_H_
