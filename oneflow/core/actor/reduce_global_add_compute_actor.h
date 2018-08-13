#ifndef ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceGlobalAddCompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAddCompActor);
  ReduceGlobalAddCompActor() = default;
  ~ReduceGlobalAddCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;

  HashMap<int64_t, std::string> regst_desc_id2bn_in_op_;
  std::tuple<std::string, bool, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_
