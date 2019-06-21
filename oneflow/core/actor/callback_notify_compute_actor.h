#ifndef ONEFLOW_CORE_ACTOR_CALLBACK_NOTIFY_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_CALLBACK_NOTIFY_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

class CallbackNotifyCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyCompActor);
  CallbackNotifyCompActor() = default;
  ~CallbackNotifyCompActor() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CALLBACK_NOTIFY_COMPUTE_ACTOR_H_
