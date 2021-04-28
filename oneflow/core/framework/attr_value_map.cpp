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
  JUST(HasAttr(attr_name));
  const auto& it = attrs_->find(attr_name);
  if (it != attrs_->end()) { return it->second; }
  if (!parents_) {
    for (const auto& p : *parents_) {
      if (JUST(p.HasAttr(attr_name))) { return p.GetAttr<cfg::AttrValue>(attr_name); }
    }
  }
  UNIMPLEMENTED_THEN_RETURN() << "The attribute with name \"" << attr_name
                              << "\" has not been found.";
}

#define DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER(field, cpp_type, attr_type)         \
  template<>                                                                       \
  Maybe<cpp_type> AttrValueMap::GetAttr(const std::string& attr_name) const {      \
    const auto& it = attrs_->find(attr_name);                                      \
    if (it != attrs_->end()) {                                                     \
      AttrValue attr_vale;                                                         \
      it->second->ToProto(&attr_vale);                                             \
      return user_op::AttrValueAccessor<cpp_type>::Attr(attr_vale);                \
    }                                                                              \
    if (!parents_) {                                                               \
      for (const auto& p : *parents_) {                                            \
        if (JUST(p.HasAttr(attr_name))) { return p.GetAttr<cpp_type>(attr_name); } \
      }                                                                            \
    }                                                                              \
    UNIMPLEMENTED_THEN_RETURN() << "The attribute with name \"" << attr_name       \
                                << "\" has not been found.";                       \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_GETTER

template<>
Maybe<void> MutableAttrValueMap::SetAttr(const std::string& attr_name,
                                         const std::shared_ptr<cfg::AttrValue>& attr_val) {
  auto it = attrs_->find(attr_name);
  if (it == attrs_->end()) {
    attrs_->emplace(attr_name, attr_val);
  } else {
    it->second = attr_val;
  }
  attr_names_->emplace(attr_name);
  return Maybe<void>::Ok();
}

#define DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER(field, cpp_type, attr_type)                      \
  template<>                                                                                    \
  Maybe<void> MutableAttrValueMap::SetAttr(const std::string& attr_name, const cpp_type& val) { \
    AttrValue attr_val;                                                                         \
    user_op::AttrValueAccessor<cpp_type>::Attr(val, &attr_val);                                 \
    SetAttr(attr_name, std::make_shared<cfg::AttrValue>(attr_val));                             \
    return Maybe<void>::Ok();                                                                   \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER, ATTR_SEQ);

#undef DEFINE_ATTR_VALUE_MAP_ATTRIBUTE_SETTER

AttrValueMap::iterator::AttrValueMapIter AttrValueMap::Find(const std::string& attr_name) const {
  const auto& it = attrs_->find(attr_name);
  if (it != attrs_->end()) { return it; }
  if (!parents_) {
    for (const auto& p : *parents_) {
      const auto& it = p.find(attr_name);
      if (it != p.end()) { return it.internal(); }
    }
  }
  return it;
}

/*static*/ Maybe<AttrValueMap> AttrValueMap::Compose(const AttrValueMap& attrs,
                                                     const AttrValueMap& parent) {
  auto current = std::make_shared<AttrValueMap>();
  current->parents_ = std::make_shared<std::vector<AttrValueMap>>();
  current->parents_->emplace_back(parent);
  current->attrs_ = attrs.attrs_;
  *current->attr_names_ = *attrs.attr_names_;
  for (const auto& it : *parent.attr_names_) { current->attr_names_->emplace(it); }
  return current;
}

Maybe<MutableAttrValueMap&> MutableAttrValueMap::Compose(const AttrValueMap& parent) {
  if (!parents_) { parents_ = std::make_shared<std::vector<AttrValueMap>>(); }
  parents_->emplace_back(parent);
  for (const auto& it : *parent.attr_names_) { attr_names_->emplace(it); }
  return *this;
}

/*static*/ Maybe<MutableAttrValueMap> MutableAttrValueMap::Compose(const AttrValueMap& attrs,
                                                                   const AttrValueMap& parent) {
  auto current = std::make_shared<MutableAttrValueMap>();
  current->parents_ = std::make_shared<std::vector<AttrValueMap>>();
  current->parents_->emplace_back(parent);
  *current->attrs_ = *attrs.attrs_;
  *current->attr_names_ = *attrs.attr_names_;
  for (const auto& it : *parent.attr_names_) { current->attr_names_->emplace(it); }
  return current;
}

}  // namespace oneflow
