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
#ifndef ONEFLOW_CORE_FRAMEWORK_CONFIG_DTR_H_
#define ONEFLOW_CORE_FRAMEWORK_CONFIG_DTR_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

Maybe<void> EnableDTRStrategy(bool enable_dtr, size_t thres, bool enable_debug, int memory_policy,
                              bool use_disjoint_set);

}  // namespace oneflow

#endif
