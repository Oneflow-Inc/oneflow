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
#ifndef ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor();

 private:
  class CommNetDeviceCtx;
  struct RegstCtx {
    void* comm_net_token;
    Regst* regst_raw_ptr;
    int64_t producer;
    int64_t act_id;
    bool has_sole_empty_blob;
  };

  void VirtualActorInit(const TaskProto&) override;
  void InitDeviceCtx(const ThreadCtx&) override;
  void SetReadableRegstInfo(const Regst*, ReadableRegstInfo*) const override;

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

  bool is_in_eord_;
  HashMap<int64_t, RegstCtx> piece_id2regst_ctx_;
  void* actor_read_id_;
  CommNetDeviceCtx* comm_net_device_ctx_;
  int64_t next_piece_id_;
  int64_t in_regst_desc_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
