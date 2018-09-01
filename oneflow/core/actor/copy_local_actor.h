#ifndef ONEFLOW_CORE_ACTOR_COPY_LOCAL_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_LOCAL_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyLocalActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyLocalActor);
  CopyLocalActor() = default;
  ~CopyLocalActor() = default;

 private:
  void VirtualActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {true, {}};
  }
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_LOCAL_ACTOR_H_
