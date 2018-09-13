#ifndef ONEFLOW_CORE_ACTOR_INPUT_WISE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_INPUT_WISE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class InputWiseCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InputWiseCompActor);
  InputWiseCompActor() = default;
  ~InputWiseCompActor() = default;

 protected:
  void Init(const TaskProto&);
  int64_t cur_processed_regst_desc_id() const { return cur_processed_regst_desc_id_; }
  int64_t processed_regst_desc_id_cnt() const { return processed_regst_desc_id_cnt_; }
  int64_t RegstDescNum() const { return readable_regsts_.size(); }
  int64_t InBnId4RegstDescId(int64_t id) const { return regst_desc_id2in_bn_id_.at(id); }
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;
  bool EnableInplace() const {
    return GetDeviceType() == DeviceType::kGPU && Global<JobDesc>::Get()->enable_mem_sharing();
  }

  bool ProducedCtrlRegstValid(const Regst* regst) const override;
  bool IsCustomizedCtrlReady() override { return true; }

 private:
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
  virtual void SetKernelCtxOther(void** other) { *other = nullptr; }
  void UpdateMemberStatusAfterAct();
  bool NeedSendRegstMsgToConsumer();
  void UpdateMemberStatusAfterSendRegstMsgToConsumer();
  virtual void VirtualUpdateMemberStatusAfterAct() {}
  virtual void VirtualUpdateMemberStatusAfterSendRegstMsgToConsumer() {}

  HashMap<int64_t, std::queue<Regst*>> readable_regsts_;
  int64_t readable_regst_desc_cnt_;
  HashMap<int64_t, bool> regst_desc_id2is_processed_;
  int64_t processed_regst_desc_id_cnt_;
  int64_t cur_processed_regst_desc_id_;
  int64_t just_processed_regst_desc_id_;

  HashMap<int64_t, int64_t> regst_desc_id2in_bn_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_INPUT_WISE_COMPUTE_ACTOR_H_
