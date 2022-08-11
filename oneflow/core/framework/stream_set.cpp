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
#include <vector>
#include <mutex>
#include "oneflow/core/framework/stream_set.h"
#include "oneflow/core/common/env_var/stream.h"

namespace oneflow {

namespace {

int64_t GetStreamSetId() {
  static std::atomic<int64_t> stream_set_id(0);
  return ++stream_set_id;
}

}  // namespace

StreamSet::StreamSet(int64_t stream_set_id) : stream_set_id_(stream_set_id) {}

StreamSet::StreamSet() : stream_set_id_(GetStreamSetId()) {}

}  // namespace oneflow
