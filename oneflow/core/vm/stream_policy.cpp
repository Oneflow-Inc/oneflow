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
#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"
#include "oneflow/core/common/env_var/vm.h"

namespace oneflow {
namespace vm {

bool StreamPolicy::OnSchedulerThread(StreamRole stream_role) const {
  if (StreamOnIndependentThread::Visit(stream_role)) { return false; }
  return ThreadLocalEnvBool<ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD>();
}

}  // namespace vm
}  // namespace oneflow
