#ifndef ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_

#include "oneflow/core/actor/accumulate_actor.h"

namespace oneflow {

class MdDiffAccActor final : public AccumulateActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccActor);
  MdDiffAccActor() = default;
  ~MdDiffAccActor() = default;

  void VirtualCompActorInit(const TaskProto& proto,
                            const ThreadCtx& ctx) override {
    AccumulateActor::Init(proto, ctx,
                          JobDesc::Singleton()->NumOfPiecesInBatch());
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_
