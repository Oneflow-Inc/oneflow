#ifndef ONEFLOW_CORE_ACTOR_SOURCE_TICK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SOURCE_TICK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class SourceTickComputeActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickComputeActor);
  SourceTickComputeActor() = default;
  ~SourceTickComputeActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SOURCE_TICK_COMPUTE_ACTOR_H_
