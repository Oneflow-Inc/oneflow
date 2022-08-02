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

namespace oneflow {

class AccActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccActor);
  AccActor() = default;
  ~AccActor() override = default;

 private:
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  void VirtualActorInit(const TaskProto& proto) override;

  int32_t acc_cnt_{};
  int32_t max_acc_cnt_{};
};

void AccActor::VirtualActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Singleton<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Singleton<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_GE(in_time_shape.elem_cnt(), out_time_shape.elem_cnt());
  max_acc_cnt_ = in_time_shape.elem_cnt() / out_time_shape.elem_cnt();
  acc_cnt_ = 0;
  OF_SET_MSG_HANDLER(&AccActor::HandlerNormal);
}

void AccActor::Act() {
  if (acc_cnt_ == 0) {
    Regst* out_regst = GetNaiveCurWriteable("out");
    Regst* in_regst = GetNaiveCurReadable("in");
    const Blob* in_blob = in_regst->GetMutSoleBlob();
    Blob* out_blob = out_regst->GetMutSoleBlob();
    const size_t size = in_blob->ByteSizeOfBlobBody();
    CHECK_EQ(out_blob->ByteSizeOfBlobBody(), size);
    AutoMemcpy(actor_ctx()->stream_ctx()->stream(), out_blob->ForceMutDptr(), in_blob->dptr(), size,
               out_blob->mem_case(), in_blob->mem_case());
  } else {
    AsyncLaunchKernel();
  }
  acc_cnt_ += 1;
}

void AccActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer();
    acc_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kAcc, AccActor);

}  // namespace oneflow
