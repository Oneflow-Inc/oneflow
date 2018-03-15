#ifndef ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_

#include "oneflow/core/actor/actor.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class RecordLoadActor final : public ActorIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadActor);
  RecordLoadActor() = default;
  ~RecordLoadActor() = default;

 private:
  void Init(const TaskProto&, const ThreadCtx&) override;

  int HandlerWaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&);
  int HandlerZombie(const ActorMsg& msg);
  int TrySwitchToZombieOrFinish();
  void TryUpdtStateAsProducedRegst(Regst* regst);
  void Act();
  void ActUntilFail();
  bool IsLoadDone();

  std::queue<std::unique_ptr<Regst>> produced_regsts_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  PbRf<int64_t> consumers_actor_id_;
  int32_t piece_id_;
  bool is_eof_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
