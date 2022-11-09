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
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

class SourceTickActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickActor);
  SourceTickActor() = default;
  ~SourceTickActor() = default;

 private:
  void VirtualActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);
};

void SourceTickActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&SourceTickActor::HandlerWaitToStart);
}

void SourceTickActor::Act() {}

bool SourceTickActor::IsCustomizedReadReady() const {
  // NOTE(chengcheng): SourceTickActor CANNOT be used and need delete in the future
  return true;
}

int SourceTickActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&SourceTickActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kSourceTick, SourceTickActor);

}  // namespace oneflow
