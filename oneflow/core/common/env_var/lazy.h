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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_

#include <string>
#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

// Default compilation mode during graph compilation. There 4 modes to choose:
// "naive", master rank compile the full plan.
// "rank_per_process", multi process(rank) run seperation compile.
// "rank_per_iter", one thread run seperation compile. Just for debug.
// "rank_per_thread", multi thread run seperation compile. Just for debug.
DEFINE_THREAD_LOCAL_ENV_STRING(ONEFLOW_LAZY_COMPILE_MODE, "naive");
// Default number of threads during graph compilation.
DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM, 16);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_
