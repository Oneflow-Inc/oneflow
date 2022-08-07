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
#ifndef ONEFLOW_CORE_FRAMEWORK_TMP_COMPUTE_STREAM_TYPE_GUARD_H_
#define ONEFLOW_CORE_FRAMEWORK_TMP_COMPUTE_STREAM_TYPE_GUARD_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/thread_local_guard.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {

class TmpComputeStreamTypeGuard final : public ThreadLocalGuard<std::string> {
 public:
  using ThreadLocalGuard<std::string>::ThreadLocalGuard;
  ~TmpComputeStreamTypeGuard() = default;

  static Maybe<Symbol<Stream>> TryConvertToTmpCompute(Symbol<Stream> stream) {
    if (!Current().has_value()) { return stream; }
    if (stream->stream_type() == StreamType::kCompute) {
      return Stream::New(stream->device(), StreamType::kTmpCompute, *JUST(Current()));
    } else if (stream->stream_type() == StreamType::kHost2Device) {
      return Stream::New(stream->device(), StreamType::kTmpHost2Device);
    } else if (stream->stream_type() == StreamType::kDevice2Host) {
      return Stream::New(stream->device(), StreamType::kTmpDevice2Host);
    } else {
      return stream;
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TMP_COMPUTE_STREAM_TYPE_GUARD_H_
