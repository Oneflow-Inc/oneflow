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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_SET_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_SET_H_

#include <unordered_map>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {

class StreamSet final {
 public:
  ~StreamSet();

  static Maybe<StreamSet> New(int64_t worker_thread_id) {
    return New(worker_thread_id, Stream::kDefaultStreamCommId);
  }

  static Maybe<StreamSet> New(int64_t worker_thread_id, int64_t comm_id);

  std::unordered_map<std::pair<Symbol<Device>, StreamType>, Symbol<Stream>>*
  mut_device_stream_type2stream() {
    return &device_stream_type2stream_;
  }

  int64_t worker_thread_id() const { return worker_thread_id_; }
  int64_t comm_id() const { return comm_id_; }

 private:
  StreamSet(int64_t worker_thread_id, int64_t comm_id);

  int64_t worker_thread_id_;
  int64_t comm_id_;
  std::unordered_map<std::pair<Symbol<Device>, StreamType>, Symbol<Stream>>
      device_stream_type2stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_SET_H_
