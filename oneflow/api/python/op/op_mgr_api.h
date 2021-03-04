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
#ifndef ONEFLOW_API_PYTHON_OP_OP_MGR_API_H_
#define ONEFLOW_API_PYTHON_OP_OP_MGR_API_H_

#include "oneflow/api/python/op/op_mgr.h"

inline bool IsOpTypeCaseCpuSupportOnly(int64_t op_type_case) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case).GetOrThrow();
}

inline bool IsOpTypeNameCpuSupportOnly(const std::string& op_type_name) {
  return oneflow::IsOpTypeNameCpuSupportOnly(op_type_name).GetOrThrow();
}

inline long long GetUserOpAttrType(const std::string& op_type_name, const std::string& attr_name) {
  return oneflow::GetUserOpAttrType(op_type_name, attr_name).GetOrThrow();
}

inline std::string InferOpConf(const std::string& serialized_op_conf,
                               const std::string& serialized_op_input_signature) {
  return oneflow::InferOpConf(serialized_op_conf, serialized_op_input_signature).GetOrThrow();
}

inline std::string GetSerializedOpAttributes() {
  return oneflow::GetSerializedOpAttributes().GetOrThrow();
}

inline bool IsInterfaceOpTypeCase(int64_t op_type_case) {
  return oneflow::IsInterfaceOpTypeCase(op_type_case).GetOrThrow();
}

inline long long GetOpParallelSymbolId(const std::string& serialized_op_conf) {
  return oneflow::GetOpParallelSymbolId(serialized_op_conf).GetOrThrow();
}

inline std::string CheckAndCompleteUserOpConf(const std::string& serialized_op_conf) {
  return oneflow::CheckAndCompleteUserOpConf(serialized_op_conf).GetOrThrow();
}

#endif  // ONEFLOW_API_PYTHON_OP_OP_MGR_API_H_
