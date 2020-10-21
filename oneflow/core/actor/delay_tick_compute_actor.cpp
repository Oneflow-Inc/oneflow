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
#include "oneflow/core/actor/delay_tick_compute_actor.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

void DelayTickCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  eord_received_ = false;
  {
    const auto& exec_node_list = task_proto.exec_sequence().exec_node();
    CHECK_EQ(exec_node_list.size(), 1);
    const auto& op_conf = exec_node_list.Get(0).kernel_conf().op_attribute().op_conf();
    CHECK(op_conf.has_delay_tick_conf());
    total_delay_num_ = op_conf.delay_tick_conf().delay_num();
  }
  TakeOverConsumedRegst(task_proto.consumed_regst_desc_id());
  TakeOverProducedRegst(task_proto.produced_regst_desc());
  OF_SET_MSG_HANDLER(&DelayTickCompActor::HandlerNormal);
}

void DelayTickCompActor::TakeOverConsumedRegst(
    const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  CHECK_EQ(consumed_ids.size(), 1);
  const auto& pair = *consumed_ids.begin();
  CHECK_EQ(pair.second.regst_desc_id_size(), 1);
  consumed_regst_desc_id_ = pair.second.regst_desc_id(0);
  consumed_rs_.InsertRegstDescId(consumed_regst_desc_id_);
  consumed_rs_.InitedDone();
}

void DelayTickCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool DelayTickCompActor::IsCustomizedReadReady() const {
  int64_t queued_size = consumed_rs_.QueueSize4RegstDescId(consumed_regst_desc_id_);
  if (eord_received_) { return queued_size > 0; }
  return queued_size > total_delay_num_;
}

void DelayTickCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> Handler) const {
  Handler(consumed_rs_.Front(consumed_regst_desc_id_));
}

void DelayTickCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(consumed_rs_.Front(consumed_regst_desc_id_)->piece_id());
    return true;
  });
}

void DelayTickCompActor::AsyncReturnCurCustomizedReadableRegst() {
  Regst* regst = consumed_rs_.Front(consumed_regst_desc_id_);
  CHECK(regst);
  AsyncSendRegstMsgToProducer(regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(consumed_regst_desc_id_));
}

void DelayTickCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  // do nothing
  // message sending is delayed into ack of downstream producer message
  // message sent in UpdtStateAsCustomizedProducedRegst
}

void DelayTickCompActor::AsyncReturnAllCustomizedReadableRegst() {
  // Messages have been sent in UpdtStateAsCustomizedProducedRegst
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

void DelayTickCompActor::NormalProcessCustomizedEordMsg(const ActorMsg&) { eord_received_ = true; }

bool DelayTickCompActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return eord_received_ && consumed_rs_.QueueSize4RegstDescId(consumed_regst_desc_id_) == 0;
}

void DelayTickCompActor::TakeOverProducedRegst(
    const PbMap<std::string, RegstDescProto>& produced_ids) {
  CHECK_EQ(produced_ids.size(), 1);
  const auto& regst_desc_proto = produced_ids.begin()->second;
  CHECK(regst_desc_proto.regst_desc_type().has_data_regst_desc());
  CHECK_EQ(regst_desc_proto.has_inplace_consumed_regst_desc_id(), false);
  produced_regst_desc_id_ = regst_desc_proto.regst_desc_id();
  produced_rs_.InsertRegstDescId(produced_regst_desc_id_);
  produced_rs_.InitedDone();
  ForEachProducedRegst([&](Regst* regst) {
    CHECK_EQ(regst->regst_desc_id(), produced_regst_desc_id_);
    CHECK_EQ(0, produced_rs_.TryPushBackRegst(regst));
  });
}

void DelayTickCompActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(regst->regst_desc_id(), produced_regst_desc_id_);
  CHECK_EQ(0, produced_rs_.TryPushBackRegst(regst));
  // release delayed consumer message
  AsyncReturnCurCustomizedReadableRegst();
}

void DelayTickCompActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  Regst* const regst = produced_rs_.Front(produced_regst_desc_id_);
  CHECK_GT(HandleRegstToConsumer(regst, [](int64_t) { return true; }), 0);
  produced_rs_.PopFrontRegsts({produced_regst_desc_id_});
}

bool DelayTickCompActor::IsCustomizedWriteReady() const { return produced_rs_.IsCurSlotReady(); }

REGISTER_ACTOR(TaskType::kDelayTick, DelayTickCompActor);

}  // namespace oneflow
