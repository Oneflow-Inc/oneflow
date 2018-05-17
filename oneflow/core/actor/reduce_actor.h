#ifndef ONEFLOW_CORE_ACTOR_REDUCE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class ReduceActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceActor);
  ReduceActor() = default;
  ~ReduceActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  void Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) override;
  bool IsReadReady() override;
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {true, {}};
  }

  int64_t consumed_regst_num_;
  int64_t processed_regst_cnt_;
  HashMap<int64_t, Regst*> regsts_in_using_;
  HashMap<int64_t, HashSet<Regst*>> regsts_used_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_ACTOR_H_
