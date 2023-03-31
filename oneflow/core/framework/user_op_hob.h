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
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_

#include <sstream>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/high_order_bool.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace user_op {

ALWAYS_INLINE inline auto HobTrue() {
  std::ostringstream string_stream;
  string_stream << "\" always true \"";
  return hob::LiteralBool<KernelRegContext>(string_stream.str(), true);
}

ALWAYS_INLINE inline auto HobFalse() {
  std::ostringstream string_stream;
  string_stream << "\" always false \"";
  return hob::LiteralBool<KernelRegContext>(string_stream.str(), false);
}

ALWAYS_INLINE inline auto HobDataType(const std::string& tensor_name, int tensor_idx) {
  std::ostringstream string_stream;
  string_stream << "data_type of tensor \'" << tensor_name << "\'";
  return hob::make_custom(
      string_stream.str(), [tensor_name, tensor_idx](const KernelRegContext& ctx) -> DataType {
        const user_op::TensorDesc* desc = ctx.TensorDesc4ArgNameAndIndex(tensor_name, tensor_idx);
        CHECK(desc != nullptr) << "key `" << tensor_name << "_" << tensor_idx << "` not found.";
        return desc->data_type();
      });
}

ALWAYS_INLINE inline auto HobInputSize(const std::string& tensor_name) {
  std::ostringstream string_stream;
  string_stream << "size of input \'" << tensor_name << "\'";
  return hob::make_custom(string_stream.str(),
                          [tensor_name](const KernelRegContext& ctx) -> int32_t {
                            return ctx.user_op_conf().input_size(tensor_name);
                          });
}

template<typename T>
ALWAYS_INLINE inline auto HobAttr(const std::string& attr_name) {
  return hob::make_custom(attr_name, [attr_name](const user_op::KernelRegContext& ctx) -> const T& {
    return ctx.Attr<T>(attr_name);
  });
}

ALWAYS_INLINE inline auto HobDeviceType() {
  return hob::make_custom(
      "device_type", [](const KernelRegContext& ctx) -> DeviceType { return ctx.device_type(); });
}

ALWAYS_INLINE inline auto HobDeviceSubTag() {
  return hob::make_custom("device_sub_tag", [](const KernelRegContext& ctx) -> const std::string& {
    return ctx.Attr<std::string>("device_sub_tag");
  });
}

ALWAYS_INLINE inline auto HobEnvBool(const std::string& env_var, bool default_value) {
  std::ostringstream string_stream;
  string_stream << "environment variable \'" << env_var << "\'";
  return hob::make_custom(string_stream.str(),
                          [env_var, default_value](const KernelRegContext& ctx) -> bool {
                            return ParseBooleanFromEnv(env_var, default_value);
                          });
}

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
