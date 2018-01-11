#ifndef ONEFLOW_CORE_ACTOR_SOURCE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SOURCE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class SourceCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceCompActor);
  SourceCompActor() = default;
  ~SourceCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerWaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override { return !IsReadReady(); }
  void AsyncReturnAllReadableRegst() override {}

  struct DataLoadStatus {
    DataLoadStatus()
        : next_col_id(0), max_col_id(-1), next_piece_id(0), is_eof(false) {}
    int64_t next_col_id;
    int64_t max_col_id;
    int64_t next_piece_id;
    bool is_eof;
  };

  DataLoadStatus data_load_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SOURCE_COMPUTE_ACTOR_H_
