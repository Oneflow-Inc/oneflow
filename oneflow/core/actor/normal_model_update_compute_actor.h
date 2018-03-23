#ifndef ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/actor/naive_readable_register_manager.h"

namespace oneflow {

class NormalMdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtCompActor);
  NormalMdUpdtCompActor() = default;
  ~NormalMdUpdtCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  void InitRegstBySendToFw(int64_t regst_desc_id);

  int HandlerInitModelAndModelTmp(const ActorMsg&);
  int HandlerSendInitialModel(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  bool IsWriteReady() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int8_t init_remaining_cnt_;
  bool is_model_diff_acc_eord_;
  NaiveReadableRegstMgr readable_regst_mgr_;
  int64_t next_model_version_id_;
  int64_t related_save_model_actor_id_;
  int64_t related_init_model_actor_id_;
  Regst* pre_model_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_
