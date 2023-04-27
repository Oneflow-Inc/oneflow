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
#pragma once

#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

DEFINE_ENV_BOOL(ONEFLOW_REMAT_DISPLAY_IN_FIRST_TIME, false);
DEFINE_ENV_BOOL(ONEFLOW_REMAT_RECORD_MEM_FRAG_RATE, true);
DEFINE_ENV_INTEGER(ONEFLOW_REMAT_GROUP_NUM, 1);
DEFINE_ENV_BOOL(ONEFLOW_REMAT_NEIGHBOR, true);
DEFINE_ENV_BOOL(ONEFLOW_REMAT_HEURISTIC_DTE, false);
DEFINE_ENV_BOOL(ONEFLOW_REMAT_HEURISTIC_DTR, false);
DEFINE_ENV_BOOL(ONEFLOW_REMAT_LOG, false);

}  // namespace oneflow
