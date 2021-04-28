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
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

AttrValueMap::AttrValueMap(std::initializer_list<AttrValueMap::value_type> init)
    : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : init) {
    attrs_->emplace(pair.first, pair.second);
  }
}

AttrValueMap::AttrValueMap(const MutableAttrValueMap& other) : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : other) { attrs_->emplace(pair.first, pair.second); }
}

AttrValueMap::AttrValueMap(const MutableCfgAttrValueMap& other) : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : other) {
    const auto& attr_value = CHECK_JUST(AttrValueUtil::ToCppAttrValue(*pair.second));
    attrs_->emplace(pair.first, attr_value);
  }
}

template<typename T>                                                                  
Maybe<const T&> AttrValueMap::GetAttr(const std::string& attr_name) const {
  const auto& it = this->find(attr_name);                                   
  CHECK_OR_RETURN(it != this->end());                                       
  const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(it->second.get());      
  CHECK_NOTNULL_OR_RETURN(ptr);                                          
  return ptr->val();             
}

template<typename T>                                                                  
Maybe<const T&> ComposedAttrValueMap::GetAttr(const std::string& attr_name) const {
  {
    const auto& it = prior_.find(attr_name);                                   
    if (it != prior_.end()) {
      const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(it->second.get());      
      CHECK_NOTNULL_OR_RETURN(ptr);                                          
      return ptr->val();             
    }
  }
  {
    const auto& it = base_.find(attr_name);                                   
    if (it != base_.end()) {
      const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(it->second.get());
      CHECK_NOTNULL_OR_RETURN(ptr);                                          
      return ptr->val();             
    }
  }
  return Error::ValueError(std::string("no attribute found. attribute name: ") + attr_name);
}

#define DEFINE_ATTR_VALUE_MAP_GET_ATTR(field, T, attr_type)    \
template Maybe<const T&> AttrValueMap::GetAttr<T>(const std::string& attr_name) const; \
template Maybe<const T&> ComposedAttrValueMap::GetAttr<T>(const std::string& attr_name) const;

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_GET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_GET_ATTR

}  // namespace oneflow
