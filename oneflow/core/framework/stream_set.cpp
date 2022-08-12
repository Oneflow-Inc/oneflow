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
#include "oneflow/core/common/env_var/stream.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

class StreamSetIdMap final {
 public:
  explicit StreamSetIdMap() {}
  ~StreamSetIdMap() = default;

  int64_t Get(int64_t worker_thread_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto* stream_set_ids = &worker_thread_id2stream_set_ids_[worker_thread_id];
    for (int i = 0; i < stream_set_ids->size() + 1; ++i) {
      if (stream_set_ids->count(i) == 0) {
        stream_set_ids->insert(i);
        return i;
      }
    }
    UNIMPLEMENTED();
  }

  Maybe<void> Put(int64_t worker_thread_id, int64_t stream_set_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto* stream_set_ids = &JUST(MapAt(worker_thread_id2stream_set_ids_, worker_thread_id));
    CHECK_OR_RETURN(stream_set_ids->erase(stream_set_id) > 0)
        << "no stream_set_id found. (current: " << stream_set_id << ").";
    return Maybe<void>::Ok();
  }

 private:
  std::map<int64_t, std::set<int64_t>> worker_thread_id2stream_set_ids_;
  std::mutex mutex_;
};

StreamSetIdMap* MutStreamSetIdMap() {
  static StreamSetIdMap map;
  return &map;
}

}  // namespace

StreamSet::StreamSet(int64_t worker_thread_id)
    : worker_thread_id_(worker_thread_id),
      stream_set_id_(MutStreamSetIdMap()->Get(worker_thread_id)) {}

StreamSet::~StreamSet() {
  MutStreamSetIdMap()->Put(worker_thread_id_, stream_set_id_).GetOrThrow();
}

}  // namespace oneflow
