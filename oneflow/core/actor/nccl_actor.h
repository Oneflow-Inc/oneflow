#ifndef ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_

#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

class NcclActor final : public NaiveActor {
 public:
  NcclActor() = default;
  ~NcclActor() override = default;

 private:
  void InitDeviceCtx(const ThreadCtx& thread_ctx) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NCCL_ACTOR_H_
