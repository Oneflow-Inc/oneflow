#ifndef ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class NormalForwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalForwardCompActor);
  NormalForwardCompActor() = default;
  ~NormalForwardCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void Act() override;
  bool IsCustomizedReadReady() override;
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() override {
    return {false, {"in"}};
  }
  std::pair<bool, std::vector<std::string>> GetNaiveProducedRegstDescName() override {
    return {false, {"out", "activation", "data_tmp", "fw_buf", "forward_model"}};
  }
  bool IsCustomizedWriteReady() override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  bool CheckOutputActId(int64_t regst_desc_id) const override;

  int HandlerInitModelAndConstBuf(const ActorMsg&);
  void UpdateModelRegstPtr(Regst* regst);
  void AsyncInitModelAndConstBuf();
  void AsyncReturnModelRegst();
  void TryAsyncReturnModelRegst();
  void TryAsyncReturnConstModelRegst();
  void TrySendMsgToForwardModelSaveActor(int64_t piece_id);
  void SendMsgToForwardModelSaveActor(int64_t batch_id);
  void SendConstBufInitMsgToBwActor();

  int64_t model_regst_desc_id_;
  int64_t const_model_regst_desc_id_;
  int64_t forward_model_regst_desc_id_;
  int64_t random_seed_;
  Regst* model_regst_;
  Regst* const_model_regst_;
  Regst* pre_forward_model_regst_;

  // customized produced
  int64_t const_buf_regst_desc_id_;
  Regst* const_buf_regst_;
  bool send_const_buf_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMAL_FORWARD_COMPUTE_ACTOR_H_
