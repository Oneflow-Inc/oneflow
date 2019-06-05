#ifndef ONEFLOW_CORE_ACTOR_SLICE_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SLICE_BOXING_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class SliceBoxingActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingActor);
  SliceBoxingActor() = default;
  ~SliceBoxingActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override { InputWiseCompActor::Init(proto); }
  void SetKernelCtxOther(void** other) override;

  std::pair<int64_t, int64_t> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SLICE_BOXING_ACTOR_H_
