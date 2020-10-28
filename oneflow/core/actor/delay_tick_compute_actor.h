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
#ifndef ONEFLOW_CORE_ACTOR_DELAY_TICK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_DELAY_TICK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class TaskProto;

class DelayTickCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DelayTickCompActor);
  DelayTickCompActor() = default;
  ~DelayTickCompActor() override = default;

 protected:
  void VirtualCompActorInit(const TaskProto&) override;

  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override { return true; }

 private:
  void Act() override;
  // consumed regst slot
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  bool IsCustomizedReadReady() const override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override;

  void AsyncReturnCurCustomizedReadableRegst();
  void AsyncReturnAllCustomizedReadableRegst() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  void TakeOverConsumedRegst(const PbMap<std::string, RegstDescIdSet>& consumed_ids);

  // produced regst slot
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  bool IsCustomizedWriteReady() const override;
  bool CheckOutputActId(int64_t regst_desc_id) const override { return false; }
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void TakeOverProducedRegst(const PbMap<std::string, RegstDescProto>& produced_ids);

  bool eord_received_;
  size_t total_delay_num_;
  int64_t consumed_regst_desc_id_;
  RegstSlot consumed_rs_;
  int64_t produced_regst_desc_id_;
  RegstSlot produced_rs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_DELAY_TICK_COMPUTE_ACTOR_H_
