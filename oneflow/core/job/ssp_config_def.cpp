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
#include "oneflow/core/framework/config_def.h"

namespace oneflow {

namespace {

REGISTER_FUNCTION_CONFIG_DEF()
    .Bool("enable_ssp", false, "enable ssp")
    .String("ssp_partition_strategy", "naive_sequantial",
            "ssp partition strategy, Avaiable strategies: naive_sequantial | disable")
    .ListInt64("ssp_partition_scope_ids", {}, "type: list[int64]. ssp partition scope symbol ids");

REGISTER_SCOPE_CONFIG_DEF()
    .Int64("ssp_num_stages", -1, "total number of ssp stages")
    .Int64("ssp_stage_id", -1, "current ssp stage id ");

}  // namespace

}  // namespace oneflow
