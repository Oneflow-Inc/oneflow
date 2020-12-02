/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
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
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
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
