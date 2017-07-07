#ifndef ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class BpDataCompActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BpDataCompActor);
  BpDataCompActor() = default;
  ~BpDataCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleNormal(const ActorMsg&) override;
  int HandleWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  int ProcessEord() override { TODO(); }

  CudaStreamHandle cuda_handle_;
  int num_of_read_empty_;
  int num_of_not_eord_;
  int64_t expected_model_version_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  std::queue<std::pair<int64_t, int64_t>> piece_model_id_;
  // <regst_desc_id, queue<regst_wp>>
  HashMap<int64_t, std::queue<std::shared_ptr<RegstWrapper>>> read_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_
