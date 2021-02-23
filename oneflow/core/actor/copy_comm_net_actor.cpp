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
#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

CopyCommNetActor::~CopyCommNetActor() { Global<CommNet>::Get()->DeleteActorReadId(actor_read_id_); }

class CopyCommNetActor::CommNetDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx() = delete;
  ~CommNetDeviceCtx() = default;

  CommNetDeviceCtx(void* actor_read_id) : actor_read_id_(actor_read_id) {}
  std::unique_ptr<DeviceCtx> Copy() const { UNIMPLEMENTED(); }

  void AddCallBack(std::function<void()> callback) const override {
    Global<CommNet>::Get()->AddReadCallBack(actor_read_id_, callback);
  }

 private:
  void* actor_read_id_;
};

void CopyCommNetActor::VirtualActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  next_piece_id_ = 0;
  in_regst_desc_id_ = Name2SoleRegstDescId("copy_in");
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

void CopyCommNetActor::InitDeviceCtx(const ThreadCtx&) {
  actor_read_id_ = Global<CommNet>::Get()->NewActorReadId();
  comm_net_device_ctx_ = new CommNetDeviceCtx(actor_read_id_);
  mut_device_ctx().reset(comm_net_device_ctx_);
}

void CopyCommNetActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(piece_id2regst_ctx_.at(next_piece_id_).regst_raw_ptr);
}

void CopyCommNetActor::SetReadableRegstInfo(const Regst* regst, ReadableRegstInfo* info) const {
  const RegstCtx& regst_ctx = piece_id2regst_ctx_.at(next_piece_id_);
  CHECK(regst == regst_ctx.regst_raw_ptr);
  info->set_regst_desc_id(in_regst_desc_id_);
  info->set_act_id(regst_ctx.act_id);
}

bool CopyCommNetActor::NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg& msg) {
  RegstCtx regst_ctx;
  regst_ctx.comm_net_token = msg.comm_net_token();
  regst_ctx.regst_raw_ptr = msg.regst();
  regst_ctx.producer = msg.src_actor_id();
  regst_ctx.act_id = msg.act_id();
  regst_ctx.has_sole_empty_tensor_in_sole_tensor_list =
      msg.has_sole_empty_tensor_in_sole_tensor_list();
  CHECK(piece_id2regst_ctx_.emplace(msg.piece_id(), regst_ctx).second);
  return true;
}

void CopyCommNetActor::Act() {
  // readable
  auto readable_it = piece_id2regst_ctx_.find(next_piece_id_);
  void* readable_token = readable_it->second.comm_net_token;
  int64_t src_actor_id = readable_it->second.producer;
  int64_t src_machine_id = Global<IDMgr>::Get()->MachineId4ActorId(src_actor_id);
  // writeable
  Regst* writeable_regst = GetNaiveCurWriteable("copy_out");
  if (readable_it->second.has_sole_empty_tensor_in_sole_tensor_list) {
    // pass if regst dynamic body is emtpy
    Blob* data_blob = writeable_regst->GetMutSoleBlob();
    TensorBackInserter back_inserter(data_blob);
    back_inserter.ReserveOneEmptyTensorList();
    FullyMutTensorView* tensor_view = back_inserter.add_tensor();
    Shape empty_shape = data_blob->static_shape();
    for (int i = 0; i < empty_shape.NumAxes(); ++i) { empty_shape.Set(i, 0); }
    tensor_view->set_shape(empty_shape);
  } else {
    void* writeable_token = writeable_regst->comm_net_token();
    // Async
    Global<CommNet>::Get()->Read(actor_read_id_, src_machine_id, readable_token, writeable_token);
  }
}

void CopyCommNetActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(next_piece_id_);
    return true;
  });
}

void CopyCommNetActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  auto readable_it = piece_id2regst_ctx_.find(next_piece_id_);
  EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToProducer(actor_id(), readable_it->second.producer,
                                                    readable_it->second.regst_raw_ptr));
  piece_id2regst_ctx_.erase(readable_it);
  next_piece_id_ += 1;
}

bool CopyCommNetActor::IsCustomizedReadReady() const {
  return piece_id2regst_ctx_.find(next_piece_id_) != piece_id2regst_ctx_.end();
}

bool CopyCommNetActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return is_in_eord_ && piece_id2regst_ctx_.empty();
}

void CopyCommNetActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK(piece_id2regst_ctx_.empty());
}

REGISTER_ACTOR(TaskType::kCopyCommNet, CopyCommNetActor);

}  // namespace oneflow
