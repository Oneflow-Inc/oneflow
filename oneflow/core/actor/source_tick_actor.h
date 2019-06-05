#ifndef ONEFLOW_CORE_ACTOR_SOURCE_TICK_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SOURCE_TICK_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class SourceTickActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickActor);
  SourceTickActor() = default;
  ~SourceTickActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
