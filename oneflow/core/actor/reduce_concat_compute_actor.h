#ifndef ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceConcatCompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatCompActor);
  ReduceConcatCompActor() = default;
  ~ReduceConcatCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;

  HashMap<int64_t, std::string> regst_desc_id2bn_in_op_;
  std::pair<int64_t, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_CONCAT_COMPUTE_ACTOR_H_
