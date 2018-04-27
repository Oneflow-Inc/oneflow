#ifndef ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class RecordLoadActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadActor);
  RecordLoadActor() = default;
  ~RecordLoadActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {false, {}};
  }
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  int32_t piece_id_;
  bool is_eof_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
