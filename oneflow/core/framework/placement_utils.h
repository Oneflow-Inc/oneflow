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
#ifndef _ONEFLOW_CORE_FRAMEWORK_PLACEMENT_UTILS_H_
#define _ONEFLOW_CORE_FRAMEWORK_PLACEMENT_UTILS_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

Maybe<Symbol<ParallelDesc>> ReplacePlacementDeviceTag(Symbol<ParallelDesc> parallel_desc,
                                                      const std::string& device_type);

Maybe<void> TouchGlobalTensor(const std::shared_ptr<one::Tensor>& tensor);

constexpr auto* CheckMetaConsistency = DECORATE(&TouchGlobalTensor, CheckGlobalTensorMeta);

}  // namespace oneflow

#endif
