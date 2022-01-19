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
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace mlir {

namespace oneflow {

namespace support {

using ::oneflow::UserOpDef;
using ::oneflow::user_op::OpRegistryResult;
using ::oneflow::user_op::UserOpRegistryMgr;

const UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const OpRegistryResult* val = UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

std::vector<std::string> GetInputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).input()) { ret.push_back(arg.name()); }
  return ret;
}

std::vector<std::string> GetOutputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).output()) { ret.push_back(arg.name()); }
  return ret;
}

}  // namespace support

}  // namespace oneflow

}  // namespace mlir
