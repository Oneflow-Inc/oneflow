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
#include "oneflow/core/actor/esac_compute_actor.h"

namespace oneflow {

void EsacCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
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
  OF_SET_MSG_HANDLER(&EsacCompActor::HandlerNormal);
}

void EsacCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool EsacCompActor::IsCustomizedReadReady() const { return -1 != GetCurProcessedRegstDescId(); }

void EsacCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(consumed_rs_.Front(cur_processed_regst_desc_id_));
}

void EsacCompActor::Act() {
  cur_processed_regst_desc_id_ = GetCurProcessedRegstDescId();
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &in_bn_id;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (cur_processed_regst_desc_id_ != regst_desc_id) { return nullptr; }
    return cur_regst;
  });
}

void EsacCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(consumed_rs_.Front(cur_processed_regst_desc_id_)->piece_id());
    return true;
  });
}

void EsacCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  AsyncSendRegstMsgToProducer(cur_regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(cur_processed_regst_desc_id_));
  cur_processed_regst_desc_id_ = -1;
}

void EsacCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool EsacCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

int64_t EsacCompActor::GetCurProcessedRegstDescId() const {
  int64_t cur_processed_regst_desc_id = -1;
  consumed_rs_.ForChosenRegstDeq(
      [&cur_processed_regst_desc_id](int64_t) { return cur_processed_regst_desc_id == -1; },
      [&cur_processed_regst_desc_id](const std::deque<Regst*>& reg_deq) {
        if (reg_deq.empty()) { return; }
        cur_processed_regst_desc_id = reg_deq.front()->regst_desc_id();
      });
  return cur_processed_regst_desc_id;
}

REGISTER_ACTOR(kEsac, EsacCompActor);

}  // namespace oneflow
