#ifndef ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/kernel/record_load_kernel.h"

namespace oneflow {

class RecordLoadActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadActor);
  RecordLoadActor() = default;
  ~RecordLoadActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_;
  bool is_eof_;
  RecordLoadStatus record_load_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
