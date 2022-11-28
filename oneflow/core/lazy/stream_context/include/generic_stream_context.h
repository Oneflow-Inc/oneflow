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
#ifndef ONEFLOW_CORE_LAZY_STREAM_CONTEXT_GENERIC_STREAM_CONTEXT_H_
#define ONEFLOW_CORE_LAZY_STREAM_CONTEXT_GENERIC_STREAM_CONTEXT_H_

#include "oneflow/core/lazy/stream_context/include/stream_context.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

class GenericStreamContext : public StreamContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenericStreamContext);
  explicit GenericStreamContext(const StreamId& stream_id);
  ~GenericStreamContext() override;

  Maybe<void> AddCallback(std::function<void()> callback) override;
  DeviceType device_type() const override { return stream_->device_type(); }

  ep::Stream* stream() override;

 private:
  ep::Stream* stream_;
  Channel<std::pair<ep::Event*, std::function<void()>>> cb_event_chan_;
  std::thread poller_thread_;
  std::shared_ptr<ep::Device> device_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_STREAM_CONTEXT_GENERIC_STREAM_CONTEXT_H_
