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
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

class CollectiveBoxingGenericActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericActor);
  CollectiveBoxingGenericActor() = default;
  ~CollectiveBoxingGenericActor() override = default;

 private:
  void Act() override { AsyncLaunchKernel(GenDefaultKernelCtx()); }

  void VirtualActorInit(const TaskProto&) override {
    piece_id_ = 0;
    OF_SET_MSG_HANDLER(&CollectiveBoxingGenericActor::HandlerNormal);
  }

  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override {
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(piece_id_);
      return true;
    });
    piece_id_ += 1;
  }

  void InitDeviceCtx(const ThreadCtx& thread_ctx) override {
    mut_device_ctx().reset(new CollectiveBoxingDeviceCtx());
  }

  int64_t piece_id_ = 0;
};

REGISTER_ACTOR(TaskType::kCollectiveBoxingGeneric, CollectiveBoxingGenericActor);

}  // namespace oneflow
