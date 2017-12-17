#ifndef ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

class MdDiffAccCompActor final : public AccumulateCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccCompActor);
  MdDiffAccCompActor() = default;
  ~MdDiffAccCompActor() = default;

  void VirtualCompActorInit(const TaskProto& proto) override {
    AccumulateCompActor::Init(proto,
                              JobDesc::Singleton()->NumOfPiecesInBatch());
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_
