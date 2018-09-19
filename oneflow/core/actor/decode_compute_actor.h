#ifndef ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

class DecodeCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeCompActor);
  DecodeCompActor() = default;
  ~DecodeCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;

  int32_t piece_id_;
  DecodeStatus decode_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_
