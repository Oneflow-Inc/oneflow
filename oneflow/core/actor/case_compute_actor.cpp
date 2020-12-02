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
#include "oneflow/core/actor/case_compute_actor.h"

namespace oneflow {

void CaseCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const int32_t output_bns_size =
      task_proto.exec_sequence().exec_node().Get(0).kernel_conf().op_attribute().output_bns_size();
  FOR_RANGE(int64_t, i, 0, output_bns_size) {
    const int64_t regst_desc_id =
        exec_kernel_vec().at(0).bn_in_op2blob_info.at(GenRepeatedBn("out", i)).regst_desc_id;
    CHECK(out_bn_id2regst_desc_id_.emplace(i, regst_desc_id).second);
  }
  TakeOverConsumedRegst(task_proto.consumed_regst_desc_id());
  TakeOverProducedRegst(task_proto.produced_regst_desc());
  OF_SET_MSG_HANDLER(&CaseCompActor::HandlerNormal);
}

void CaseCompActor::TakeOverConsumedRegst(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  CHECK_EQ(consumed_ids.size(), 1);
  const auto& pair = *consumed_ids.begin();
  CHECK_EQ(pair.second.regst_desc_id_size(), 1);
  consumed_regst_desc_id_ = pair.second.regst_desc_id(0);
  consumed_rs_.InsertRegstDescId(consumed_regst_desc_id_);
  consumed_rs_.InitedDone();
}

void CaseCompActor::TakeOverProducedRegst(const PbMap<std::string, RegstDescProto>& produced_ids) {
  for (const auto& pair : produced_ids) {
    CHECK(pair.second.regst_desc_type().has_data_regst_desc());
    CHECK_EQ(pair.second.has_inplace_consumed_regst_desc_id(), false);
    const int64_t regst_desc_id = pair.second.regst_desc_id();
    regst_desc_id2produced_rs_[regst_desc_id].InsertRegstDescId(regst_desc_id);
    regst_desc_id2produced_rs_.at(regst_desc_id).InitedDone();
  }
  ForEachProducedRegst([&](Regst* regst) {
    const int64_t regst_desc_id = regst->regst_desc_id();
    CHECK_EQ(0, regst_desc_id2produced_rs_.at(regst_desc_id).TryPushBackRegst(regst));
  });
}

// twice called for each output
// first called: set cur_selected_id
// second called: output cur_selected_id
void CaseCompActor::Act() {
  Regst* const consumed_regst = consumed_rs_.Front(consumed_regst_desc_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  case_status_.cur_selected_id = GetCurSelectId();
  case_status_.cmd =
      (case_status_.cur_selected_id == -1 ? kCaseCmdHandleInput : kCaseCmdHandleOutput);
  kernel_ctx.other = &case_status_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (consumed_regst_desc_id_ == regst_desc_id) { return consumed_regst; }
    return regst_desc_id2produced_rs_.at(regst_desc_id).Front(regst_desc_id);
  });
}

void CaseCompActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  const int64_t regst_desc_id = regst->regst_desc_id();
  CHECK_EQ(0, regst_desc_id2produced_rs_.at(regst_desc_id).TryPushBackRegst(regst));
}

bool CaseCompActor::IsCustomizedReadReady() const { return IsInputOrOutputReady(); }

bool CaseCompActor::IsCustomizedWriteReady() const { return IsInputOrOutputReady(); }

bool CaseCompActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return ReceiveEordMsg(consumed_regst_desc_id_) && case_status_.select_id2request_cnt.size() == 0;
}

bool CaseCompActor::IsInputOrOutputReady() const {
  if (GetCurSelectId() != -1) { return true; }
  return consumed_rs_.IsCurSlotReady();
}

int64_t CaseCompActor::GetCurSelectId() const {
  for (const auto& pair : case_status_.select_id2request_cnt) {
    CHECK_GT(pair.second, 0);
    const int64_t regst_desc_id = out_bn_id2regst_desc_id_.at(pair.first);
    if (regst_desc_id2produced_rs_.at(regst_desc_id).IsCurSlotReady()) { return pair.first; }
  }
  return -1;
}

void CaseCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> Handler) const {
  Handler(consumed_rs_.Front(consumed_regst_desc_id_));
}

void CaseCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (case_status_.cmd != kCaseCmdHandleInput) { return; }
  Regst* const cur_regst = consumed_rs_.Front(consumed_regst_desc_id_);
  CHECK_NOTNULL(cur_regst);
  AsyncSendRegstMsgToProducer(cur_regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(consumed_regst_desc_id_));
}

void CaseCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

void CaseCompActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  if (case_status_.cmd != kCaseCmdHandleOutput) { return; }
  const int64_t regst_desc_id = out_bn_id2regst_desc_id_.at(case_status_.cur_selected_id);
  Regst* const regst = regst_desc_id2produced_rs_.at(regst_desc_id).Front(regst_desc_id);
  CHECK_GT(HandleRegstToConsumer(regst, [](int64_t) { return true; }), 0);
  regst_desc_id2produced_rs_.at(regst_desc_id).PopFrontRegsts({regst_desc_id});
}

bool CaseCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

bool CaseCompActor::CheckOutputActId(int64_t regst_desc_id) const { return false; }

REGISTER_ACTOR(kCase, CaseCompActor);

}  // namespace oneflow
