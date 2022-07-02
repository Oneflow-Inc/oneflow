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
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor();

 private:
  struct RegstCtx {
    void* comm_net_token;
    Regst* regst_raw_ptr;
    int64_t producer;
    bool has_sole_empty_blob;
  };

  void VirtualActorInit(const TaskProto&) override;

  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override { is_in_eord_ = true; }
  bool NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg&) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override;
  void AsyncReturnAllCustomizedReadableRegst() override;
  void AddCallback(std::function<void()> callback) override;
  bool is_in_eord_;
  HashMap<int64_t, RegstCtx> sequence_number2regst_ctx_;
  void* actor_read_id_;
  int64_t next_sequence_number_;
  int64_t in_regst_desc_id_;
};

CopyCommNetActor::~CopyCommNetActor() {
  Singleton<CommNet>::Get()->DeleteActorReadId(actor_read_id_);
}

void CopyCommNetActor::VirtualActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  next_sequence_number_ = 0;
  in_regst_desc_id_ = Name2SoleRegstDescId("copy_in");
  actor_read_id_ = Singleton<CommNet>::Get()->NewActorReadId();
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

void CopyCommNetActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(sequence_number2regst_ctx_.at(next_sequence_number_).regst_raw_ptr);
}

bool CopyCommNetActor::NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg& msg) {
  RegstCtx regst_ctx;
  regst_ctx.comm_net_token = msg.comm_net_token();
  regst_ctx.regst_raw_ptr = msg.regst();
  regst_ctx.producer = msg.src_actor_id();
  regst_ctx.has_sole_empty_blob = msg.has_sole_empty_blob();
  CHECK(sequence_number2regst_ctx_.emplace(msg.comm_net_sequence_number(), regst_ctx).second);
  return true;
}

void CopyCommNetActor::Act() {
  // readable
  auto readable_it = sequence_number2regst_ctx_.find(next_sequence_number_);
  void* readable_token = readable_it->second.comm_net_token;
  int64_t src_actor_id = readable_it->second.producer;
  int64_t src_machine_id = MachineId4ActorId(src_actor_id);
  // writeable
  Regst* writeable_regst = GetNaiveCurWriteable("copy_out");
  if (readable_it->second.has_sole_empty_blob) {
    // pass if regst dynamic body is emtpy
    Blob* data_blob = writeable_regst->GetMutSoleBlob();
    Shape empty_shape = data_blob->static_shape();
    for (int i = 0; i < empty_shape.NumAxes(); ++i) { empty_shape.Set(i, 0); }
    data_blob->mut_shape_view()->set_shape(empty_shape);
  } else {
    void* writeable_token = writeable_regst->comm_net_token();
    // Async
    Singleton<CommNet>::Get()->Read(actor_read_id_, src_machine_id, readable_token,
                                    writeable_token);
  }
}

void CopyCommNetActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
}

void CopyCommNetActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  auto readable_it = sequence_number2regst_ctx_.find(next_sequence_number_);
  EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToProducer(actor_id(), readable_it->second.producer,
                                                    readable_it->second.regst_raw_ptr));
  sequence_number2regst_ctx_.erase(readable_it);
  next_sequence_number_ += 1;
}

bool CopyCommNetActor::IsCustomizedReadReady() const {
  return sequence_number2regst_ctx_.find(next_sequence_number_) != sequence_number2regst_ctx_.end();
}

bool CopyCommNetActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return is_in_eord_ && sequence_number2regst_ctx_.empty();
}

void CopyCommNetActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK(sequence_number2regst_ctx_.empty());
}

void CopyCommNetActor::AddCallback(std::function<void()> callback) {
  Singleton<CommNet>::Get()->AddReadCallBack(actor_read_id_, callback);
}

REGISTER_ACTOR(TaskType::kCopyCommNet, CopyCommNetActor);

}  // namespace oneflow
