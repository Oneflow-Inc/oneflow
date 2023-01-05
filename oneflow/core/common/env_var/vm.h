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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_VM_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_VM_H_

#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD, true);
DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_VM_ENABLE_STREAM_WAIT, true);
DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_VM_PENDING_HANDLE_WINDOW_SIZE, 10)
DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_VM_ENABLE_SCHEDULE_YIELD, true)
DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_VM_WORKER_THREAD_LIMIT, 16);
DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_VM_MULTI_THREAD, true);

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_VM_H_
