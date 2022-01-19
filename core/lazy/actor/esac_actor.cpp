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
#include "oneflow/core/lazy/actor/actor.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/kernel/esac_kernel.h"

namespace oneflow {

class EsacActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EsacActor);
  EsacActor() = default;
  ~EsacActor() override = default;

 protected:
  void VirtualActorInit(const TaskProto&) override;
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
  int64_t cur_processed_regst_desc_id_{};
  HashMap<int64_t, int64_t> regst_desc_id2in_bn_id_;
};

void EsacActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const int32_t input_bns_size =
      task_proto.exec_sequence().exec_node().Get(0).kernel_conf().op_attribute().input_bns_size();
  FOR_RANGE(int64_t, i, 0, input_bns_size) {
    const int64_t regst_desc_id =
        exec_kernel_vec().at(0).bn_in_op2blob_info.at(GenRepeatedBn("in", i)).regst_desc_id;
    CHECK(regst_desc_id2in_bn_id_.emplace(regst_desc_id, i).second);
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (const int64_t regst_desc_id : pair.second.regst_desc_id()) {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
    }
  }
  consumed_rs_.InitedDone();
  cur_processed_regst_desc_id_ = -1;
  OF_SET_MSG_HANDLER(&EsacActor::HandlerNormal);
}

void EsacActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool EsacActor::IsCustomizedReadReady() const { return -1 != GetCurProcessedRegstDescId(); }

void EsacActor::ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)> handler) const {
  handler(consumed_rs_.Front(cur_processed_regst_desc_id_));
}

void EsacActor::Act() {
  cur_processed_regst_desc_id_ = GetCurProcessedRegstDescId();
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id_);
  CHECK_EQ(exec_kernel_vec().size(), 1);
  CHECK_NOTNULL(dynamic_cast<EsacKernelState*>(exec_kernel_vec().at(0).kernel_ctx->state().get()))
      ->value = in_bn_id;
  AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* {
    if (cur_processed_regst_desc_id_ != regst_desc_id) { return nullptr; }
    return cur_regst;
  });
}

void EsacActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
}

void EsacActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  AsyncSendRegstMsgToProducer(cur_regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(cur_processed_regst_desc_id_));
  cur_processed_regst_desc_id_ = -1;
}

void EsacActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool EsacActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

int64_t EsacActor::GetCurProcessedRegstDescId() const {
  int64_t cur_processed_regst_desc_id = -1;
  consumed_rs_.ForChosenRegstDeq(
      [&cur_processed_regst_desc_id](int64_t) { return cur_processed_regst_desc_id == -1; },
      [&cur_processed_regst_desc_id](const std::deque<Regst*>& reg_deq) {
        if (reg_deq.empty()) { return; }
        cur_processed_regst_desc_id = reg_deq.front()->regst_desc_id();
      });
  return cur_processed_regst_desc_id;
}

REGISTER_ACTOR(kEsac, EsacActor);

}  // namespace oneflow
