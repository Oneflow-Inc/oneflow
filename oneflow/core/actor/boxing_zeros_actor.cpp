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
#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

class BoxingZerosActor : public NaiveActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingZerosActor);
  BoxingZerosActor() = default;
  ~BoxingZerosActor() override = default;

  void VirtualActorInit(const TaskProto& task_proto) override {
    NaiveActor::VirtualActorInit(task_proto);
    piece_id_ = 0;
    out_inited_ = false;
  }

 private:
  void Act() override {
    if (!out_inited_) {
      NaiveActor::Act();
      out_inited_ = true;
    }
  }

  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override {
    int64_t piece_id = piece_id_;
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(piece_id);
      return true;
    });
    piece_id_ += 1;
  }

  int64_t piece_id_;
  bool out_inited_;
};

REGISTER_ACTOR(TaskType::kBoxingZeros, BoxingZerosActor);

}  // namespace oneflow
