#ifndef ONEFLOW_CORE_ACTOR_OUT_STREAM_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_OUT_STREAM_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/sink_compute_actor.h"
#include "oneflow/core/actor/of_serving.h"

namespace oneflow {

class OutStreamCompActor final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStreamCompActor);
  OutStreamCompActor() = default;
  ~OutStreamCompActor() = default;

 private:
  void Act() override;
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void VirtualSinkCompActorInit(const TaskProto&) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_OUT_STREAM_COMPUTE_ACTOR_H_