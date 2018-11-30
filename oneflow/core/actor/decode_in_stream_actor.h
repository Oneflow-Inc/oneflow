#ifndef ONEFLOW_CORE_ACTOR_DECODE_IN_STREAM_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DECODE_IN_STREAM_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/actor/of_serving.h"

namespace oneflow {

class DecodeInStreamActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeInStreamActor);
  DecodeInStreamActor() = default;
  ~DecodeInStreamActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return {RegstNameType::kNaive, {}};
  }
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_ = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DECODE_IN_STREAM_ACTOR_H_
