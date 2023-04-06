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
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

int64_t* MutThreadLocalUniqueGlobalId() {
  static thread_local int64_t global_id = kThreadGlobalIdMain;
  return &global_id;
}

}  // namespace

int64_t GetThisThreadGlobalId() { return *MutThreadLocalUniqueGlobalId(); }

ThreadGlobalIdGuard::ThreadGlobalIdGuard(int64_t thread_global_id)
    : old_thread_global_id_(GetThisThreadGlobalId()) {
  if (old_thread_global_id_ != kThreadGlobalIdMain) {
    CHECK_EQ(old_thread_global_id_, thread_global_id)
        << "nested ThreadGlobalIdGuard disabled. old thread_global_id: " << old_thread_global_id_
        << ", new thread_global_id:" << thread_global_id;
  }
  *MutThreadLocalUniqueGlobalId() = thread_global_id;
}

ThreadGlobalIdGuard::~ThreadGlobalIdGuard() {
  *MutThreadLocalUniqueGlobalId() = old_thread_global_id_;
}

}  // namespace oneflow
