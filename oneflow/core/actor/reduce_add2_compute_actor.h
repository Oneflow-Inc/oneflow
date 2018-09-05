#ifndef ONEFLOW_CORE_ACTOR_REDUCE_ADD2_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_ADD2_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceAdd2CompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceAdd2CompActor);
  ReduceAdd2CompActor() = default;
  ~ReduceAdd2CompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;

  HashMap<int64_t, std::string> regst_desc_id2bn_in_op_;
  std::tuple<std::string, bool, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_ADD2_COMPUTE_ACTOR_H_
