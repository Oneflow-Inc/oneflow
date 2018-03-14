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

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override {}

  bool is_in_eord_;
  std::queue<Regst*> pending_in_regsts_;
  DecodeStatus decode_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DECODE_COMPUTE_ACTOR_H_
