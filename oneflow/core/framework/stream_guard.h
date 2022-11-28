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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_GUARD_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_GUARD_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/env_var/stream.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_set.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/thread/thread_global_id.h"

namespace oneflow {

class StreamConverter final {
 public:
  explicit StreamConverter(const std::shared_ptr<StreamSet>& stream_set)
      : stream_set_(stream_set) {}

  Maybe<Symbol<Stream>> TryConvertStream(Symbol<Stream> stream) {
    size_t thread_uid = stream_set_->worker_thread_id();
    return Stream::New(stream->device(), stream->stream_type(), thread_uid);
  }

 private:
  const std::shared_ptr<StreamSet> stream_set_;
};

class StreamGuard final {
 public:
  explicit StreamGuard(const std::shared_ptr<StreamConverter>& stream_converter) {
    old_value_ = Current();
    *MutCurrent() = stream_converter;
  }
  ~StreamGuard() { *MutCurrent() = old_value_; }

  static Maybe<Symbol<Stream>> TryConvertStream(Symbol<Stream> stream) {
    if (!Current().has_value()) { return stream; }
    return JUST(Current())->TryConvertStream(stream);
  }

 private:
  static const Optional<StreamConverter>& Current() { return *MutCurrent(); }
  static Optional<StreamConverter>* MutCurrent();

  Optional<StreamConverter> old_value_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GUARD_H_
