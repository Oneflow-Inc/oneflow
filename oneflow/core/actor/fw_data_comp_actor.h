#ifndef ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class FwDataCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FwDataCompActor);
  FwDataCompActor() = default;
  ~FwDataCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int WaitToStart(const ActorMsg&);
  int HandleNormal(const ActorMsg&) override;
  int HandleWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  int ProcessEord() override { TODO(); }

  CudaStreamHandle cuda_handle_;
  int num_of_not_eord_;
  int64_t expected_model_version_id_;
  int64_t in_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  std::shared_ptr<RegstWrapper> model_regst_;
  std::shared_ptr<RegstWrapper> model_tmp_regst_;
  std::queue<std::shared_ptr<RegstWrapper>> in_;
  HashMap<int64_t, std::shared_ptr<RegstWrapper>> ready_in_regst_;
  KernelCtx kernel_ctx_;
  int64_t bp_actor_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
