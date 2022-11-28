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
#ifndef ONEFLOW_CORE_LAZY_STREAM_CONTEXT_STREAM_CONTEXT_H_
#define ONEFLOW_CORE_LAZY_STREAM_CONTEXT_STREAM_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class StreamContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamContext);
  StreamContext() = default;
  virtual ~StreamContext() = default;

  virtual ep::Stream* stream() = 0;
  virtual Maybe<void> AddCallback(std::function<void()> callback) = 0;
  virtual DeviceType device_type() const = 0;
};

#define REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(device, creator) \
  REGISTER_CLASS_CREATOR(int, device, StreamContext, creator, const StreamId&)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_STREAM_CONTEXT_STREAM_CONTEXT_H_
