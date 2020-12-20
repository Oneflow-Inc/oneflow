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

inline std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> IsOpTypeCaseCpuSupportOnly(
    int64_t op_type_case) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case).GetDataAndErrorProto(false);
}

inline std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> IsOpTypeNameCpuSupportOnly(
    const std::string& op_type_name) {
  return oneflow::IsOpTypeNameCpuSupportOnly(op_type_name).GetDataAndErrorProto(false);
}

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> GetUserOpAttrType(
    const std::string& op_type_name, const std::string& attr_name) {
  return oneflow::GetUserOpAttrType(op_type_name, attr_name).GetDataAndErrorProto(0LL);
}

inline std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> InferOpConf(
    const std::string& serialized_op_conf, const std::string& serialized_op_input_signature) {
  return oneflow::InferOpConf(serialized_op_conf, serialized_op_input_signature)
      .GetDataAndErrorProto(std::string(""));
}

inline std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
GetSerializedOpAttributes() {
  return oneflow::GetSerializedOpAttributes().GetDataAndErrorProto(std::string(""));
}

inline std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> IsInterfaceOpTypeCase(
    int64_t op_type_case) {
  return oneflow::IsInterfaceOpTypeCase(op_type_case).GetDataAndErrorProto(false);
}

inline std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> GetOpParallelSymbolId(
    const std::string& serialized_op_conf) {
  return oneflow::GetOpParallelSymbolId(serialized_op_conf).GetDataAndErrorProto(0LL);
}

inline std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> CheckAndCompleteUserOpConf(
    const std::string& serialized_op_conf) {
  return oneflow::CheckAndCompleteUserOpConf(serialized_op_conf)
      .GetDataAndErrorProto(std::string(""));
}

#endif  // ONEFLOW_API_PYTHON_OP_OP_MGR_API_H_
