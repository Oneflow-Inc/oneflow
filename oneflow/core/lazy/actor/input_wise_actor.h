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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_INPUT_WISE_ACTOR_H_
#define ONEFLOW_CORE_LAZY_ACTOR_INPUT_WISE_ACTOR_H_

#include "oneflow/core/lazy/actor/actor.h"

namespace oneflow {

class InputWiseActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InputWiseActor);
  InputWiseActor() = default;
  ~InputWiseActor() = default;

  using Actor::Init;

 protected:
  void Init(const TaskProto&);
  int64_t cur_processed_regst_desc_id() const { return cur_processed_regst_desc_id_; }
  int64_t processed_regst_desc_id_cnt() const { return processed_regst_desc_id_cnt_; }
  int64_t RegstDescNum() const { return consumed_rs_.total_regst_desc_cnt(); }
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
  HashMap<int64_t, bool> regst_desc_id2is_processed_;
  int64_t processed_regst_desc_id_cnt_;
  int64_t cur_processed_regst_desc_id_;

  HashMap<int64_t, int64_t> regst_desc_id2in_bn_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_INPUT_WISE_ACTOR_H_
