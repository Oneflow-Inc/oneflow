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
#include "oneflow/core/job/placement_scope.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

Maybe<Symbol<ParallelDesc>> PlacementScope::GetParallelDesc(const std::string& device_tag,
                                                            const OperatorConf& op_conf) const {
  if (device_tag == "cpu" || IsCpuOnly(op_conf)) {
    return host_parallel_desc_;
  } else {
    return device_parallel_desc_;
  }
}

Maybe<Symbol<ParallelDesc>> PlacementScope::GetParallelDesc(const std::string& device_tag,
                                                            const std::string& op_type_name) const {
  if (device_tag == "cpu" || IsCpuOnly(op_type_name)) {
    return host_parallel_desc_;
  } else {
    return device_parallel_desc_;
  }
}

Maybe<Symbol<ParallelDesc>> PlacementScope::GetParallelDesc(const std::string& op_type_name) const {
  return GetParallelDesc(device_parallel_desc_->device_tag(), op_type_name);
}

}  // namespace oneflow
