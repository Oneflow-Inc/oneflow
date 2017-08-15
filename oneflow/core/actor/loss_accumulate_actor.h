#ifndef ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_ACTOR_H_

#include "oneflow/core/actor/accumulate_actor.h"

namespace oneflow {

class LossAccActor final : public AccumulateActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccActor);
  LossAccActor() = default;
  ~LossAccActor() = default;

  void Init(const TaskProto& proto, const ThreadCtx& ctx) override {
    AccumulateActor::Init(proto, ctx,
                          JobDesc::Singleton()->piece_num_of_record_loss());
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_LOSS_ACCUMULATE_ACTOR_H_
