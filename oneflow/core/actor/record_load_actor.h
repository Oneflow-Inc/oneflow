#ifndef ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

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
  void TryUpdtStateAsProducedRegst(Regst* regst);
  void Act();
  void ActUntilFail();

  std::vector<std::unique_ptr<Regst>> produced_regsts_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  std::vector<int64_t> consumers_actor_ids_;
  std::string data_path_;
  RecordType record_type_;
  int32_t piece_id_;
  bool is_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
