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

#include "oneflow/core/framework/attr_map_util.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

AttrMap MakeAttrMap(const UserOpConf& user_conf) {
  const auto& attrs =
      std::make_shared<HashMap<std::string, std::shared_ptr<const user_op::AttrVal>>>();
  for (const auto& kv : user_conf.attr()) {
    attrs->emplace(kv.first, CHECK_JUST(user_op::AttrValueUtil::ToCppAttrValue(kv.second)));
  }
  return AttrMap(attrs);
}

}  // namespace oneflow
