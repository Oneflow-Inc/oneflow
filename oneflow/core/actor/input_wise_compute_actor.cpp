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
#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

void InputWiseCompActor::Init(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const auto& input_bns =
      task_proto.exec_sequence().exec_node().Get(0).kernel_conf().op_attribute().input_bns();
  HashMap<std::string, int64_t> ibn2in_bn_id;
  for (int64_t i = 0; i < input_bns.size(); ++i) {
    CHECK(ibn2in_bn_id.emplace(input_bns.Get(i), i).second);
  }
  for (const auto& pair : exec_kernel_vec().at(0).bn_in_op2blob_info) {
    auto it = ibn2in_bn_id.find(pair.first);
    if (it != ibn2in_bn_id.end()) {
      CHECK(regst_desc_id2in_bn_id_.emplace(pair.second.regst_desc_id, it->second).second);
    }
  }

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
      CHECK(regst_desc_id2is_processed_.emplace(regst_desc_id, false).second);
    }
  }
  consumed_rs_.InitedDone();
  cur_processed_regst_desc_id_ = -1;
  processed_regst_desc_id_cnt_ = 0;
  OF_SET_MSG_HANDLER(&InputWiseCompActor::HandlerNormal);
}

int64_t InputWiseCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id2in_bn_id_.size();
}

void InputWiseCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool InputWiseCompActor::IsCustomizedReadReady() const {
  return -1 != GetCurProcessedRegstDescId();
}

void InputWiseCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(consumed_rs_.Front(cur_processed_regst_desc_id_));
}

void InputWiseCompActor::Act() {
  cur_processed_regst_desc_id_ = GetCurProcessedRegstDescId();
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);

  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (cur_processed_regst_desc_id_ != regst_desc_id) { return nullptr; }
    return cur_regst;
  });
  processed_regst_desc_id_cnt_ += 1;
  regst_desc_id2is_processed_.at(cur_processed_regst_desc_id_) = true;
}

void InputWiseCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (processed_regst_desc_id_cnt_ == regst_desc_id2is_processed_.size()) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(consumed_rs_.Front(cur_processed_regst_desc_id_)->piece_id());
      return true;
    });
    for (auto& pair : regst_desc_id2is_processed_) {
      CHECK(pair.second);
      pair.second = false;
    }
    processed_regst_desc_id_cnt_ = 0;
  }
}

void InputWiseCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  AsyncSendRegstMsgToProducer(cur_regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(cur_processed_regst_desc_id_));
  cur_processed_regst_desc_id_ = -1;
}

void InputWiseCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, processed_regst_desc_id_cnt_);
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool InputWiseCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

int64_t InputWiseCompActor::GetCurProcessedRegstDescId() const {
  int64_t cur_processed_regst_desc_id = -1;
  consumed_rs_.ForChosenRegstDeq(
      [cur_processed_regst_desc_id](int64_t) { return cur_processed_regst_desc_id == -1; },
      [this, &cur_processed_regst_desc_id](const std::deque<Regst*>& reg_deq) {
        if (reg_deq.empty()) { return; }
        int64_t regst_desc_id = reg_deq.front()->regst_desc_id();
        if (regst_desc_id2is_processed_.at(regst_desc_id) == false) {
          cur_processed_regst_desc_id = regst_desc_id;
        }
      });
  return cur_processed_regst_desc_id;
}

}  // namespace oneflow
