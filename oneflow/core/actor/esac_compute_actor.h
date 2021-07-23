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
