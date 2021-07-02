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

#include "oneflow/core/framework/user_op_conf_trait.h"

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {
namespace user_op {

UserOpConfTrait::UserOpConfTrait(const std::string& op_name, const UserOpConf& proto)
    : op_name_(op_name), op_type_name_(proto.op_type_name()) {
  for (const auto& kv : proto.input()) {
    auto it = inputs_.emplace(kv.first, std::vector<std::string>{}).first;
    for (const auto& input : kv.second.s()) { it->second.emplace_back(input); }
  }
  for (const auto& kv : proto.output()) {
    auto it = outputs_.emplace(kv.first, std::vector<std::string>{}).first;
    for (const auto& output : kv.second.s()) { it->second.emplace_back(output); }
  }
  for (const auto& kv : proto.attr()) {
    AttrValue::ValueCase value_case = kv.second.value_case();
    switch (value_case) {
#define CASE_ENTRY(field, cpp_type, attr_type)                                      \
  /* AttrValue::ValueCase has the same order and naming convention as AttrType */   \
  case (static_cast<AttrValue::ValueCase>(attr_type)):                              \
    CHECK(attrs_                                                                    \
              .emplace(kv.first, std::make_shared<TypedAttrVal<cpp_type>>(          \
                                     AttrValueAccessor<cpp_type>::Attr(kv.second))) \
              .second);                                                             \
    break;
      OF_PP_FOR_EACH_TUPLE(CASE_ENTRY, ATTR_SEQ)
#undef CASE_ENTRY
      default: LOG(FATAL) << "Wrong attr value type: " << static_cast<int32_t>(value_case);
    };
  }
}

}  // namespace user_op
}  // namespace oneflow
