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
#include <set>
#include <map>
#include "oneflow/core/framework/stream_set.h"
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/common/env_var/stream.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

StreamSet::StreamSet(int64_t worker_thread_id) : worker_thread_id_(worker_thread_id) {}

StreamSet::~StreamSet() {}

/*static*/ Maybe<StreamSet> StreamSet::New(int64_t worker_thread_id) {
  return std::shared_ptr<StreamSet>(new StreamSet(worker_thread_id));
}

}  // namespace oneflow
