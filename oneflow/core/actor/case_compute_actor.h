#ifndef ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"
#include "oneflow/core/kernel/case_kernel.h"

namespace oneflow {

class CaseCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseCompActor);
  CaseCompActor() = default;
  ~CaseCompActor() override = default;

 protected:
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedWriteReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst);
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  void VirtualCompActorInit(const TaskProto&) override;
  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override {}
  bool CheckOutputActId(int64_t regst_desc_id) const override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedProducedRegstDescName() override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }

 private:
  void Act() override;
  void TakeOverConsumedRegst(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void TakeOverProducedRegst(const PbMap<std::string, RegstDescProto>& produced_ids);
  bool IsInputOrOutputReady() const;
  int64_t GetCurSelectId() const;

  HashMap<int64_t, int64_t> regst_desc_id2piece_id_;
  HashMap<int64_t, int64_t> out_bn_id2regst_desc_id_;
  int64_t consumed_regst_desc_id_;
  RegstSlot consumed_rs_;
  HashMap<int64_t, RegstSlot> regst_desc_id2produced_rs_;
  CaseStatus case_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_
