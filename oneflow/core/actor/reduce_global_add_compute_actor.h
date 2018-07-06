#ifndef ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class ReduceGlobalAddCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGlobalAddCompActor);
  ReduceGlobalAddCompActor() = default;
  ~ReduceGlobalAddCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  bool IsCustomizedReadReady() override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() override {
    return ReceiveAllEordMsg() && readable_regst_desc_cnt_ == 0;
  }
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {false, {}};
  }

  HashMap<int64_t, std::string> regst_desc_id2bn_in_op_;
  HashMap<int64_t, std::queue<Regst*>> readable_regsts_;
  int64_t readable_regst_desc_cnt_;
  HashSet<int64_t> unprocessed_regst_desc_id_;
  int64_t cur_processed_regst_desc_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_GLOBAL_ADD_COMPUTE_ACTOR_H_
