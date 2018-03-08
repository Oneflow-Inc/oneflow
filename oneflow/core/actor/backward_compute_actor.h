#ifndef ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class BackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardCompActor);
  BackwardCompActor() = default;
  ~BackwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;
  void AsyncReturnModelRegstUntilMatchCurOutRegst();
  void AsyncReturnModelRegstUntilLastPieceIdGreaterThan(int64_t piece_id);
  void Act() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t activation_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  int64_t out_regst_desc_id_;
  int64_t out_diff_regst_desc_id_;
  bool is_out_diff_eord_;
  HashMap<int64_t, std::queue<Regst*>> readable_regsts_;
  int64_t readable_regst_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BACKWARD_COMPUTE_ACTOR_H_
