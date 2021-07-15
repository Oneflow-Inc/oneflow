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
#ifndef ONEFLOW_CORE_FRAMEWORK_PRARLLEL_CONF_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_PRARLLEL_CONF_UTIL_H_

#include <utility>
#include <vector>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape.cfg.h"

namespace oneflow {

Maybe<std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>>
GetDeviceTagAndMachineDeviceIdsAndHierarchy(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf);

Maybe<cfg::ParallelConf> MakeParallelConf(const std::string& device_tag,
                                          const std::vector<std::string>& machine_device_ids,
                                          const std::shared_ptr<Shape>& hierarchy);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PRARLLEL_CONF_UTIL_H_
