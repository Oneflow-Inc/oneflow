#ifndef ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CompActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompActor);
  virtual ~CompActor() = default;

 protected:
  CompActor() = default;

  virtual void VirtualCompActorInit(const TaskProto& task_proto) {}

 private:
  void VirtualActorInit(const TaskProto& task_proto) override { VirtualCompActorInit(task_proto); }
};

inline int64_t GetLastPieceIdForModelVersionId(int32_t staleness, int64_t model_version_id) {
  CHECK_GE(staleness, 0);
  if (staleness == -1) { return std::numeric_limits<int64_t>::max(); }
  int32_t num_of_pieces_in_batch = Global<JobDesc>::Get()->NumOfPiecesInBatch();
  return (model_version_id + staleness + 1) * num_of_pieces_in_batch - 1;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
