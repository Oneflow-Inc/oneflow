#ifndef ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class FwDataCompActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(FwDataCompActor);
  FwDataCompActor() = default;
  ~FwDataCompActor() = default;

  void Init(const TaskProto&) override;
  int ProcessMsg(const ActorMsg&, const ThreadContext&) override;

private:
  bool IsReadReady();
  void WardKernelAndSendMsg(const KernelCtx&);

  uint64_t expected_model_version_id_;
  uint64_t model_regst_desc_id_;
  uint64_t model_tmp_regst_desc_id_;
  std::shared_ptr<RegstWarpper> model_regst_;
  std::shared_ptr<RegstWarpper> model_tmp_regst_;
  std::queue<std::shared_ptr<RegstWarpper>> in_;
  HashMap<uint64_t, std::shared_ptr<RegstWarpper>> ready_in_regst_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
