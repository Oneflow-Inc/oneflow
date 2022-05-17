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
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

size_t HashAttrName2AttrValWrapper(const oneflow::AttrName2AttrValWrapper& attr_name2attr_val) {
  size_t hash_value = 0;
  for (const auto& pair : attr_name2attr_val) {
    hash_value ^= std::hash<std::string>()(pair.first);
    hash_value ^= pair.second->hash_value();
  }
  return hash_value;
}

const AttrName2AttrValWrapper& EmptyAttrName2AttrVal() {
  static const auto empty = std::make_shared<AttrName2AttrVal>();
  static const AttrName2AttrValWrapper empty_symbol(empty);
  return empty_symbol;
}

}  // namespace

AttrName2AttrValWrapper::AttrName2AttrValWrapper(
    const std::shared_ptr<const AttrName2AttrVal>& attrs)
    : attrs_(attrs) {
  hash_value_ = HashAttrName2AttrValWrapper(*this);
}

bool AttrName2AttrValWrapper::operator==(const AttrName2AttrValWrapper& other) const {
  if (this->size() != other.size()) { return false; }
  for (const_iterator this_iter = this->begin(), that_iter = other.begin();
       this_iter != this->end(); ++this_iter, ++that_iter) {
    if (this_iter->first != that_iter->first) { return false; }
    if (*this_iter->second != *that_iter->second) { return false; }
  }
  return true;
}

AttrMap::AttrMap() : attrs_(EmptyAttrName2AttrVal()) {}

AttrMap::AttrMap(const std::shared_ptr<const AttrName2AttrVal>& attrs) : attrs_(attrs) {}

namespace {

AttrName2AttrValWrapper MakeAttrName2AttrValWrapper(
    const std::initializer_list<AttrMap::value_type>& init) {
  const auto& attrs = std::make_shared<AttrName2AttrVal>();
  for (const auto& pair : init) { attrs->emplace(pair.first, pair.second); }
  return AttrName2AttrValWrapper(attrs);
}

AttrName2AttrValWrapper MakeAttrName2AttrValWrapper(const MutableAttrMap& other) {
  const auto& attrs = std::make_shared<AttrName2AttrVal>();
  for (const auto& pair : other) { attrs->emplace(pair.first, pair.second); }
  return AttrName2AttrValWrapper(attrs);
}

AttrName2AttrValWrapper MakeAttrName2AttrValWrapper(const MutableCfgAttrMap& other) {
  const auto& attrs = std::make_shared<AttrName2AttrVal>();
  for (const auto& pair : other) {
    const auto& attr_value = CHECK_JUST(user_op::AttrValueUtil::ToCppAttrValue(*pair.second));
    attrs->emplace(pair.first, attr_value);
  }
  return AttrName2AttrValWrapper(attrs);
}

}  // namespace

AttrMap::AttrMap(std::initializer_list<AttrMap::value_type> init)
    : attrs_(MakeAttrName2AttrValWrapper(init)) {}

AttrMap::AttrMap(const MutableAttrMap& other) : attrs_(MakeAttrName2AttrValWrapper(other)) {}

AttrMap::AttrMap(const MutableCfgAttrMap& other) : attrs_(MakeAttrName2AttrValWrapper(other)) {}

AttrMap& AttrMap::operator=(const AttrMap& other) {
  attrs_ = other.attrs_;
  return *this;
}

bool AttrMap::operator==(const AttrMap& other) const { return attrs_ == other.attrs_; }

template<typename T>
Maybe<const T&> AttrMap::GetAttr(const std::string& attr_name) const {
  const auto& it = this->find(attr_name);
  CHECK_OR_RETURN(it != this->end()) << attr_name << " not found";
  const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(it->second.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  return ptr->val();
}

const std::shared_ptr<const user_op::AttrVal>& AttrMap::Attr4Name(
    const std::string& attr_name) const {
  const auto& iter = find(attr_name);
  if (iter != end()) { return iter->second; }
  static const std::shared_ptr<const user_op::AttrVal> none;
  return none;
}

AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf) {
  const auto& attrs = std::make_shared<AttrName2AttrVal>();
  for (const auto& kv : user_op_conf.attr()) {
    auto cpp_attr_value = user_op::AttrValueUtil::ToCppAttrValue(kv.second);
    if (cpp_attr_value.IsOk()) {
      attrs->emplace(kv.first, CHECK_JUST(cpp_attr_value));
    } else {
      LOG(ERROR) << user_op_conf.DebugString()
                 << " failed to convert to cpp attr value, key: " << kv.first;
    }
  }
  return AttrMap(attrs);
}

template<typename T>
Maybe<const T&> ComposedAttrMap::GetAttr(const std::string& attr_name) const {
  const auto& attr = Attr4Name(attr_name);
  CHECK_NOTNULL_OR_RETURN(attr.get())
      << Error::InvalidValueError(std::string("no attribute found. attribute name: ") + attr_name);
  return dynamic_cast<const user_op::TypedAttrVal<T>*>(attr.get())->val();
}

const std::shared_ptr<const user_op::AttrVal>& ComposedAttrMap::Attr4Name(
    const std::string& attr_name) const {
  const auto& prior_iter = prior_.find(attr_name);
  if (prior_iter != prior_.end()) { return prior_iter->second; }
  const auto& base_iter = base_.find(attr_name);
  if (base_iter != base_.end()) { return base_iter->second; }
  static const std::shared_ptr<const user_op::AttrVal> none;
  return none;
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
                                       const std::shared_ptr<AttrValue>& attr_val) {
  (*this)[attr_name] = attr_val;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MutableCfgAttrMap::SetAttr(const std::string& attr_name, const T& attr_val) {
  std::shared_ptr<AttrValue> proto_attr_val = std::make_shared<AttrValue>();
  user_op::AttrValueAccessor<T>::Attr(attr_val, proto_attr_val.get());
  (*this)[attr_name] = proto_attr_val;
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
