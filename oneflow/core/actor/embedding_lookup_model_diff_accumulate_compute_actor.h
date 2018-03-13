#ifndef ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

class EmbeddingLookupMdDiffAccCompActor final : public AccumulateCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupMdDiffAccCompActor);
  EmbeddingLookupMdDiffAccCompActor() = default;
  ~EmbeddingLookupMdDiffAccCompActor() = default;

  void VirtualCompActorInit(const TaskProto& proto) override {
    AccumulateCompActor::Init(proto, JobDesc::Singleton()->NumOfPiecesInBatch(),
                              ColIdOrder::kAscending);
  }

 private:
  void VirtualLaunchKernel(Regst* in_regst, Regst* out_regst,
                           KernelCtx) override;
  int32_t do_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_COMPUTE_ACTOR_H_
