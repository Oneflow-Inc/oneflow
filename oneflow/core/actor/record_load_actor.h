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

  int HandlerWaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override { return !IsReadReady(); }
  void AsyncReturnAllReadableRegst() override {}

  int32_t piece_id_;
  bool is_eof_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RECORD_LOAD_ACTOR_H_
