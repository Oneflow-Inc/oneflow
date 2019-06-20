#ifndef ONEFLOW_CORE_ACTOR_ESAC_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ESAC_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class EsacCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EsacCompActor);
  EsacCompActor() = default;
  ~EsacCompActor() override = default;

 protected:
  void VirtualCompActorInit(const TaskProto&) override;
  int64_t InBnId4RegstDescId(int64_t id) const { return regst_desc_id2in_bn_id_.at(id); }

  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  bool IsCustomizedReadReady() const override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override {
    return ReceiveAllEordMsg() && consumed_rs_.available_regst_desc_cnt() == 0;
  }
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  int64_t GetCurProcessedRegstDescId() const;

  RegstSlot consumed_rs_;
  int64_t cur_processed_regst_desc_id_;
  HashMap<int64_t, int64_t> regst_desc_id2in_bn_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ESAC_COMPUTE_ACTOR_H_
