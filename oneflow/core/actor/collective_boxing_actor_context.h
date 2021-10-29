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
#ifndef ONEFLOW_CORE_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_

#include "oneflow/core/actor/actor_context.h"
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

class CollectiveBoxingActorContext : public ActorContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingActorContext);
  CollectiveBoxingActorContext() = default;
  ~CollectiveBoxingActorContext() override = default;

  void Init(const TaskProto& task_proto, StreamContext* stream_ctx) override {
    stream_ctx_ = stream_ctx;
    task_proto_ = task_proto;
    collective_boxing_device_ctx_.reset(new CollectiveBoxingDeviceCtx());
  }
  void AddCallBack(std::function<void()> callback) const override {
    collective_boxing_device_ctx_->AddCallBack(std::move(callback));
  }

  CollectiveBoxingDeviceCtx* collective_boxing_device_ctx() const {
    return collective_boxing_device_ctx_.get();
  }

  StreamContext* stream_ctx() const override { return stream_ctx_; }
  const TaskProto& task_proto() const override { return task_proto_; }

 private:
  StreamContext* stream_ctx_{};
  TaskProto task_proto_{};
  std::unique_ptr<CollectiveBoxingDeviceCtx> collective_boxing_device_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
