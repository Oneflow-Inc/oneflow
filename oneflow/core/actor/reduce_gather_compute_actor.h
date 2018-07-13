#ifndef ONEFLOW_CORE_ACTOR_REDUCE_GATHER_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_GATHER_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceGatherCompActor : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGatherCompActor);
  ReduceGatherCompActor() = default;
  ~ReduceGatherCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override { InputWiseCompActor::Init(proto); }
  void SetKernelCtxOther(void** other) override;

  std::string other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_GATHER_COMPUTE_ACTOR_H_
