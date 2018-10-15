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

inline int64_t GetLastPieceIdForModelVersionId(int64_t model_version_id,
                                               int64_t num_of_pieces_in_batch) {
  return (model_version_id + 1) * num_of_pieces_in_batch - 1;
}

inline int64_t GetModelVersionIdFromPieceId(int64_t piece_id, int64_t num_of_pieces_in_batch) {
  return piece_id / num_of_pieces_in_batch;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COMPUTE_ACTOR_H_
