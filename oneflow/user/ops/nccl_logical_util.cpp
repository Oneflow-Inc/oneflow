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

#include "oneflow/user/ops/nccl_logical_util.h"

namespace oneflow {

std::string GetCommKeyFromNcclType(const std::string& op_type_name) {
  if (op_type_name == "_nccl_logical_2D_same_dim0_all_reduce"
      || op_type_name == "_nccl_logical_2D_same_dim0_all_gather"
      || op_type_name == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous"
      || op_type_name == "_nccl_logical_2D_same_dim0_all2all") {
    return "SameDim0";
  }
  if (op_type_name == "_nccl_logical_2D_same_dim1_all_reduce") { return "SameDim1"; }
  return "";
}

}  // namespace oneflow
