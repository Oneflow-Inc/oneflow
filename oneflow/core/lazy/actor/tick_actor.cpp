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
#include "oneflow/core/lazy/actor/naive_actor.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

class TickActor final : public NaiveActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TickActor);
  TickActor() = default;
  ~TickActor() = default;

 private:
  void Act() override {
    OF_PROFILER_RANGE_PUSH(std::string("tick-") + exec_kernel_vec().at(0).kernel->op_conf().name());
    OF_PROFILER_RANGE_POP();
  }
};

REGISTER_ACTOR(kTick, TickActor);
REGISTER_ACTOR(kDeviceTick, TickActor);
REGISTER_ACTOR(kSrcSubsetTick, TickActor);
REGISTER_ACTOR(kDstSubsetTick, TickActor);

}  // namespace oneflow
