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

StreamSet::StreamSet(int64_t worker_thread_id, int64_t comm_id)
    : worker_thread_id_(worker_thread_id), comm_id_(comm_id) {}

StreamSet::~StreamSet() {}

/*static*/ Maybe<StreamSet> StreamSet::New(int64_t worker_thread_id, int64_t comm_id) {
  std::shared_ptr<StreamSet> stream_set(new StreamSet(worker_thread_id, comm_id));
  constexpr int kCommIdLimit = 4;
  CHECK_GE_OR_RETURN(comm_id, 0) << "comm_id should be in range [0, " << kCommIdLimit << ")";
  CHECK_LT_OR_RETURN(comm_id, kCommIdLimit)
      << "comm_id should be in range [0, " << kCommIdLimit << ")";
  JUST(CheckWorkerThreadThreadGlobalId(comm_id));
  return stream_set;
}

}  // namespace oneflow
