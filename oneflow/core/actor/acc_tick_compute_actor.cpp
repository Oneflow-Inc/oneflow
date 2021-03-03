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
#include "oneflow/core/actor/acc_tick_compute_actor.h"

namespace oneflow {

void AccTickCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Global<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_EQ(in_time_shape.elem_cnt() % out_time_shape.elem_cnt(), 0);

  acc_cnt_ = 0;
  max_acc_cnt_ = in_time_shape.elem_cnt() / out_time_shape.elem_cnt();
  OF_SET_MSG_HANDLER(&AccTickCompActor::HandlerNormal);
}

int64_t AccTickCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id == Name2SoleRegstDescId("out") ? max_acc_cnt_ : 1;
}

void AccTickCompActor::Act() { acc_cnt_ += 1; }

void AccTickCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer();
    acc_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kAccTick, AccTickCompActor);

}  // namespace oneflow
