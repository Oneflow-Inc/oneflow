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
#include "oneflow/core/common/thread_local_guard.h"
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
    auto key = std::make_pair(stream->device(), stream->stream_type());
    auto* map = stream_set_->mut_device_stream_type2stream();
    const auto& iter = map->find(key);
    if (iter != map->end()) { return iter->second; }
    Symbol<Stream> ret;
    if (IsCommNetStream::Visit(stream->stream_type())) {
      size_t thread_uid = stream_set_->worker_thread_id();
      int64_t comm_id = stream_set_->comm_id();
      ret = JUST(Stream::New(stream->device(), stream->stream_type(), thread_uid, comm_id));
    } else {
      size_t thread_uid = stream_set_->worker_thread_id();
      ret = JUST(Stream::New(stream->device(), stream->stream_type(), thread_uid));
    }
    CHECK_OR_RETURN(map->emplace(key, ret).second) << "illegal memory access";
    return ret;
  }

 private:
  const std::shared_ptr<StreamSet> stream_set_;
};

class StreamGuard final : public ThreadLocalGuard<StreamConverter> {
 public:
  using ThreadLocalGuard<StreamConverter>::ThreadLocalGuard;
  ~StreamGuard() = default;

  static Maybe<Symbol<Stream>> TryConvertStream(Symbol<Stream> stream) {
    if (!Current().has_value()) { return stream; }
    return JUST(Current())->TryConvertStream(stream);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GUARD_H_
