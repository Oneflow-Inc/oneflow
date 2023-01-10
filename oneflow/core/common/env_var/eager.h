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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_EAGER_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_EAGER_H_

#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

// NOTE: use env variable 'ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE' indicate whether the
// use infer cache in naive local op interpret.
DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE, true);

// NOTE: use env variable 'ONEFLOW_EAGER_TENSOR_INFER_CACHE_SIZE' indicate the size of
// infer cache in op interpret.
DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_EAGER_TENSOR_INFER_CACHE_SIZE, 128 * 1024);

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_EAGER_H_
