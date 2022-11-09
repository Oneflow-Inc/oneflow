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

class AccTickActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccTickActor);
  AccTickActor() = default;
  virtual ~AccTickActor() = default;

 protected:
  void VirtualActorInit(const TaskProto& proto) override;

 private:
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  int32_t acc_cnt_;
  int32_t max_acc_cnt_;
};

void AccTickActor::VirtualActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Singleton<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Singleton<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_EQ(in_time_shape.elem_cnt() % out_time_shape.elem_cnt(), 0);

  acc_cnt_ = 0;
  max_acc_cnt_ = in_time_shape.elem_cnt() / out_time_shape.elem_cnt();
  OF_SET_MSG_HANDLER(&AccTickActor::HandlerNormal);
}

void AccTickActor::Act() { acc_cnt_ += 1; }

void AccTickActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer();
    acc_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kAccTick, AccTickActor);

}  // namespace oneflow
