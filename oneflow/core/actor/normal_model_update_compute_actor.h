#ifndef ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class NormalMdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtCompActor);
  NormalMdUpdtCompActor() = default;
  ~NormalMdUpdtCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{"const_model"});
  }
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override {}
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool IsCustomizedWriteReady() const override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void SendConstModelRegstToConsumer();
  bool CheckOutputActId(int64_t regst_desc_id) const override;
  void InitModelAndConstBuf();
  void InitRegstBySendToFw(Regst* regst);
  int HandlerInitModelAndConstModel(const ActorMsg&);
  int HandlerSendInitialModel(const ActorMsg&);
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;

  int64_t model_regst_desc_id_;
  int8_t init_remaining_cnt_;
  int64_t next_model_version_id_;
  HashSet<int64_t> related_save_model_actor_ids_;
  int64_t related_init_model_actor_id_;
  Regst* forward_model_regst_;
  Regst* pre_model_regst_;

  // customized produce
  int64_t const_model_regst_desc_id_;
  Regst* const_model_regst_;
  bool send_const_model_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMAL_MODEL_UPDATE_COMPUTE_ACTOR_H_
