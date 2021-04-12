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

#include "oneflow/core/framework/attr_value_map.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

template<>
Maybe<cfg::AttrValue> AttrValueMap::GetAttr(const std::string& attr_name) const {
  const auto& it = this->find(attr_name);
  CHECK_OR_RETURN(it != this->end());
  return it->second;
}

template<>
Maybe<void> AttrValueMap::SetAttr(const std::string& attr_name,
                                  const std::shared_ptr<cfg::AttrValue>& attr_val) {
  auto it = this->find(attr_name);
  if (it == this->end()) {
    this->emplace(attr_name, attr_val);
  } else {
    it->second = attr_val;
  }
  return Maybe<void>::Ok();
}

#define DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER(field, cpp_type, attr_type)    \
  template<>                                                                  \
  Maybe<cpp_type> AttrValueMap::GetAttr(const std::string& attr_name) const { \
    const auto& it = this->find(attr_name);                                   \
    CHECK_OR_RETURN(it != this->end());                                       \
    AttrValue attr_vale;                                                      \
    it->second->ToProto(&attr_vale);                                          \
    return user_op::AttrValueAccessor<cpp_type>::Attr(attr_vale);             \
  }

#define DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER(field, cpp_type, attr_type)               \
  template<>                                                                             \
  Maybe<void> AttrValueMap::SetAttr(const std::string& attr_name, const cpp_type& val) { \
    AttrValue attr_val;                                                                  \
    user_op::AttrValueAccessor<cpp_type>::Attr(val, &attr_val);                          \
    SetAttr(attr_name, std::make_shared<cfg::AttrValue>(attr_val));                      \
    return Maybe<void>::Ok();                                                            \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER, ATTR_SEQ);
OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER, ATTR_SEQ);

#undef DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER
#undef DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER

}  // namespace oneflow
