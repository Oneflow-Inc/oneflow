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
#ifndef ONEFLOW_API_COMMON_VARIABLE_TENSOR_MGR_H_
#define ONEFLOW_API_COMMON_VARIABLE_TENSOR_MGR_H_

#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

namespace oneflow {

inline Maybe<void> FillVariableTensorMgr(
    const std::vector<std::string>& variable_op_names,
    const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
  auto mgr = Singleton<VariableTensorMgr>::Get();
  return mgr->Fill(variable_op_names, variable_tensors);
}

inline void ResetVariableTensorMgr() {
  auto mgr = Singleton<VariableTensorMgr>::Get();
  mgr->Reset();
}

inline std::tuple<std::vector<std::string>, std::vector<std::shared_ptr<one::Tensor>>>
DumpVariableTensorMgr() {
  auto mgr = Singleton<VariableTensorMgr>::Get();
  return mgr->Dump();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_COMMON_VARIABLE_TENSOR_MGR_H_
