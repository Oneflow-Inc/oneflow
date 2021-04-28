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

#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_TRAIT_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_TRAIT_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value_map.h"

namespace oneflow {
namespace user_op {

class UserOpConfTrait {
 public:
  explicit UserOpConfTrait(const std::string& op_name, const UserOpConf& proto);
  virtual ~UserOpConfTrait() = default;

  const std::string& op_name() const { return op_name_; }
  const std::string& op_type_name() const { return op_type_name_; }

  Maybe<int> input_size(const std::string& arg_name) const {
    const auto& it = inputs_.find(arg_name);
    CHECK_OR_RETURN(it != inputs_.end());
    return it->second.size();
  }
  Maybe<int> output_size(const std::string& arg_name) const {
    const auto& it = outputs_.find(arg_name);
    CHECK_OR_RETURN(it != outputs_.end());
    return it->second.size();
  }

  template<typename T>
  Maybe<T> GetAttr(const std::string& attr_name) const {
    const auto& it = attrs_.find(attr_name);
    CHECK_OR_RETURN(it != attrs_.end()) << "The op has no attribute named " << attr_name;
    return std::dynamic_pointer_cast<TypedAttrVal<T>>(it->second)->val();
  }

  template<typename T>
  Maybe<T> GetAttr(const std::string& attr_name, const AttrValueMap& priority_attrs) const {
    if (JUST(priority_attrs.HasAttr(attr_name))) { return priority_attrs.GetAttr<T>(attr_name); }
    return GetAttr<T>(attr_name);
  }

 private:
  std::string op_name_;
  std::string op_type_name_;
  HashMap<std::string, std::vector<std::string>> inputs_;
  HashMap<std::string, std::vector<std::string>> outputs_;
  HashMap<std::string, std::shared_ptr<AttrVal>> attrs_;
};

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_TRAIT_H_
