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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_DTR_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_DTR_H_

#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

DEFINE_ENV_BOOL(OF_DTR, false);
DEFINE_ENV_BOOL(OF_DTR_EE, true);
DEFINE_ENV_BOOL(OF_DTR_LR, true);
DEFINE_ENV_BOOL(OF_DTR_O_ONE, true);
DEFINE_ENV_BOOL(OF_DTR_ALLO, true);
DEFINE_ENV_BOOL(OF_DTR_NLR, false);
DEFINE_ENV_BOOL(OF_DTR_HIGH_CONV, false);
DEFINE_ENV_BOOL(OF_DTR_HIGH_ADD_N, false);
DEFINE_ENV_BOOL(OF_DTR_CHECK, false);
DEFINE_ENV_BOOL(ONEFLOW_DTR_OLD_IMMUTABLE, false);
DEFINE_ENV_BOOL(ONEFLOW_DTR_FBIP, true);
DEFINE_ENV_BOOL(ONEFLOW_DTR_OPERATION_LOG, false);
DEFINE_ENV_BOOL(ONEFLOW_DTR_SMALL_PIECE, true);
DEFINE_ENV_BOOL(ONEFLOW_DTR_DISPLAY_IN_FIRST_TIME, false);
DEFINE_ENV_BOOL(ONEFLOW_DTR_RECORD_MEM_FRAG_RATE, true);
DEFINE_ENV_BOOL(ONEFLOW_DTR_MEGENGINE_STYLE, false);
DEFINE_ENV_BOOL(ONEFLOW_DTR_HEURISTIC_WITH_SIZE, false);
DEFINE_ENV_INTEGER(ONEFLOW_DTR_GROUP_NUM, 1);

}  // namespace oneflow

#endif
