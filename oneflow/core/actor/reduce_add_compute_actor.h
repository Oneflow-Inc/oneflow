#ifndef ONEFLOW_CORE_ACTOR_REDUCE_ADD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_ADD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceAddCompActor : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceAddCompActor);
  ReduceAddCompActor() = default;
  ~ReduceAddCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override { InputWiseCompActor::Init(proto); }
  void SetKernelCtxOther(void** other) override;

  std::pair<int64_t, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_ADD_COMPUTE_ACTOR_H_
