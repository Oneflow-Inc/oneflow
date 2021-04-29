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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

AttrMap::AttrMap(std::initializer_list<AttrMap::value_type> init) : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : init) { attrs_->emplace(pair.first, pair.second); }
}

AttrMap::AttrMap(const MutableAttrMap& other) : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : other) { attrs_->emplace(pair.first, pair.second); }
}

AttrMap::AttrMap(const MutableCfgAttrMap& other) : attrs_(new AttrName2AttrVal) {
  for (const auto& pair : other) {
    const auto& attr_value = CHECK_JUST(user_op::AttrValueUtil::ToCppAttrValue(*pair.second));
    attrs_->emplace(pair.first, attr_value);
  }
}

template<typename T>
Maybe<const T&> AttrMap::GetAttr(const std::string& attr_name) const {
  const auto& it = this->find(attr_name);
  CHECK_OR_RETURN(it != this->end());
  const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(it->second.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  return ptr->val();
}

template<typename T>
Maybe<const T&> ComposedAttrMap::GetAttr(const std::string& attr_name) const {
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

#define DEFINE_ATTR_VALUE_MAP_GET_ATTR(field, T, attr_type)                         \
  template Maybe<const T&> AttrMap::GetAttr<T>(const std::string& attr_name) const; \
  template Maybe<const T&> ComposedAttrMap::GetAttr<T>(const std::string& attr_name) const;

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_GET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_GET_ATTR

template<>
Maybe<void> MutableAttrMap::SetAttr(const std::string& attr_name,
                                    const std::shared_ptr<user_op::AttrVal>& attr_val) {
  (*this)[attr_name] = attr_val;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MutableAttrMap::SetAttr(const std::string& attr_name, const T& attr_val) {
  (*this)[attr_name] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
  return Maybe<void>::Ok();
}

template<>
Maybe<void> MutableCfgAttrMap::SetAttr(const std::string& attr_name,
                                       const std::shared_ptr<cfg::AttrValue>& attr_val) {
  (*this)[attr_name] = attr_val;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MutableCfgAttrMap::SetAttr(const std::string& attr_name, const T& attr_val) {
  AttrValue proto_attr_val;
  user_op::AttrValueAccessor<T>::Attr(attr_val, &proto_attr_val);
  (*this)[attr_name] = std::make_shared<cfg::AttrValue>(proto_attr_val);
  return Maybe<void>::Ok();
}

#define DEFINE_ATTR_VALUE_MAP_SET_ATTR(field, T, attr_type)                        \
  template Maybe<void> MutableAttrMap::SetAttr<T>(const std::string& attr_name,    \
                                                  const T& attr_val);              \
  template Maybe<void> MutableCfgAttrMap::SetAttr<T>(const std::string& attr_name, \
                                                     const T& attr_val);

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_SET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_SET_ATTR

}  // namespace oneflow
