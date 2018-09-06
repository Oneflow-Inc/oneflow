#ifndef ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD2_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD2_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceGlobalAdd2CompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAdd2CompActor);
  ReduceGlobalAdd2CompActor() = default;
  ~ReduceGlobalAdd2CompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;

  std::tuple<int64_t, bool, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD2_COMPUTE_ACTOR_H_
