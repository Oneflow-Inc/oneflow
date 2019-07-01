#ifndef ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class DecodeRandomActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomActor);
  DecodeRandomActor() = default;
  ~DecodeRandomActor() = default;

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

#endif  // ONEFLOW_CORE_ACTOR_DECODE_RANDOM_COMPUTE_ACTOR_H_
