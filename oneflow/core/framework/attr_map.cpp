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
#include "oneflow/core/framework/cached_attr_map.h"

namespace oneflow {

AttrMap::AttrMap() : data_(std::make_shared<AttrMap::AttrData>()) {}

AttrMap::AttrMap(const CachedMutableAttrMap& other) : data_(std::make_shared<AttrMap::AttrData>()) {
  data_->capacity = other.size();
  data_->hash_value = other.hash_value();
  data_->attr_names = other.attr_names();
  data_->attrs.resize(data_->capacity);
  for (int i = 0; i < data_->capacity; ++i) {
    data_->attrs[i].second = other.valid_masks()[i];
    if (other.valid_masks()[i]) {
      ++(data_->size);
      data_->attrs[i].first = other.attrs()[i];
    }
  }
}

AttrMap::AttrMap(const UserOpConf& user_op_conf) : data_(std::make_shared<AttrMap::AttrData>()) {
  data_->attr_names.reset(new small_vector<std::string, kInitializedSize>());
  for (const auto& kv : user_op_conf.attr()) {
    auto cpp_attr_value = user_op::AttrValueUtil::ToCppAttrValue(kv.second);
    if (cpp_attr_value.IsOk()) {
      ++(data_->size);
      data_->attr_names->emplace_back(kv.first);
      data_->attrs.emplace_back(CHECK_JUST(cpp_attr_value), true);

      HashCombine(&data_->hash_value, kv.first.size());
      HashCombine(&data_->hash_value, data_->attrs.back().first->hash_value());
    } else {
      LOG(ERROR) << user_op_conf.DebugString()
                 << " failed to convert to cpp attr value, key: " << kv.first;
    }
  }
  data_->capacity = data_->size;
}

AttrMap& AttrMap::operator=(const AttrMap& other) {
  data_ = other.data_;
  return *this;
}

bool AttrMap::operator==(const AttrMap& other) const {
  if (data_->size != other.data_->size || data_->hash_value != other.data_->hash_value) {
    return false;
  }
  for (int i = 0; i < std::min(data_->size, other.data_->size); ++i) {
    if (data_->attrs[i].second != other.data_->attrs[i].second) { return false; }
    if (data_->attrs[i].second) {
      if ((*data_->attr_names)[i] != (*other.data_->attr_names)[i]) { return false; }
      if (*(data_->attrs[i].first) != *(other.data_->attrs[i].first)) { return false; }
    }
  }
  return true;
}

template<typename T>
Maybe<const T&> AttrMap::GetAttr(const std::string& attr_name) const {
  const auto& attr = Attr4Name(attr_name);
  CHECK_OR_RETURN(attr) << Error::InvalidValueError()
                        << "no attribute found. attribute name: " << attr_name;
  const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(attr.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  return ptr->val();
}

const std::shared_ptr<const user_op::AttrVal>& AttrMap::Attr4Name(
    const std::string& attr_name) const {
  for (int i = 0; i < data_->capacity; ++i) {
    if (data_->attrs[i].second && attr_name == (*data_->attr_names)[i]) {
      return data_->attrs[i].first;
    }
  }
  static const std::shared_ptr<const user_op::AttrVal> none;
  return none;
}

bool AttrMap::HasAttr4Name(const std::string& attr_name) const {
  return Attr4Name(attr_name) != nullptr;
}

AttrMap::const_iterator::const_iterator(size_t pos, const AttrMap::AttrData* data)
    : pos_(pos), data_(data) {
  while (pos_ < data_->capacity) {
    if (data_->attrs[pos_].second) { break; }
    ++pos_;
  }
  if (pos_ < data_->capacity) {
    kv_.first = (*data_->attr_names)[pos_];
    kv_.second = data_->attrs[pos_].first;
  }
}

AttrMap::const_iterator& AttrMap::const_iterator::operator++() {
  ++pos_;
  while (pos_ < data_->capacity) {
    if (data_->attrs[pos_].second) { break; }
    ++pos_;
  }
  if (pos_ < data_->capacity) {
    kv_.first = (*data_->attr_names)[pos_];
    kv_.second = data_->attrs[pos_].first;
  }
  return *this;
}

AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf) { return AttrMap(user_op_conf); }

template<typename T>
Maybe<const T&> ComposedAttrMap::GetAttr(const std::string& attr_name) const {
  const auto& attr = Attr4Name(attr_name);
  CHECK_OR_RETURN(attr) << Error::InvalidValueError()
                        << "no attribute found. attribute name: " << attr_name;
  return dynamic_cast<const user_op::TypedAttrVal<T>*>(attr.get())->val();
}

const std::shared_ptr<const user_op::AttrVal>& ComposedAttrMap::Attr4Name(
    const std::string& attr_name) const {
  const auto& prior_attr = prior_.Attr4Name(attr_name);
  if (prior_attr) { return prior_attr; }
  return base_.Attr4Name(attr_name);
}

bool ComposedAttrMap::HasAttr4Name(const std::string& attr_name) const {
  return Attr4Name(attr_name) != nullptr;
}

#define DEFINE_ATTR_VALUE_MAP_GET_ATTR(field, T, attr_type)                         \
  template Maybe<const T&> AttrMap::GetAttr<T>(const std::string& attr_name) const; \
  template Maybe<const T&> ComposedAttrMap::GetAttr<T>(const std::string& attr_name) const;

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_GET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_GET_ATTR

}  // namespace oneflow
