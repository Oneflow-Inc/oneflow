#ifndef ONEFLOW_CORE_ACTOR_DATA_LOAD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DATA_LOAD_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/kernel/data_load_kernel.h"

namespace oneflow {

class DataLoadActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoadActor);
  DataLoadActor() = default;
  ~DataLoadActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  // void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_;
  bool is_eof_;
  DataLoadStatus data_load_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DATA_LOAD_ACTOR_H_
