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
#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

class AccCompActor final : public AccumulateCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompActor);
  AccCompActor() = default;
  ~AccCompActor() override = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
};

void AccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  const Shape& acc_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("acc"))
                                    .data_regst_time_shape();
  CHECK_GE(one_time_shape.elem_cnt(), acc_time_shape.elem_cnt());
  AccumulateCompActor::Init(proto, one_time_shape.elem_cnt() / acc_time_shape.elem_cnt());
}

REGISTER_ACTOR(TaskType::kAcc, AccCompActor);

}  // namespace oneflow
