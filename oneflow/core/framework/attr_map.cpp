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

constexpr int AttrMap::kInitializedSize;

AttrMap::AttrMap() : internal_(std::make_shared<AttrMap::>()) {}

AttrMap::AttrMap(const CachedMutableAttrMap& other) : internal_(std::make_shared<AttrMap::>()) {
  internal_->capacity = other.size();
  internal_->hash_value = other.hash_value();
  internal_->attr_names = other.attr_names();
  internal_->attrs.resize(internal_->capacity);
  for (int i = 0; i < internal_->capacity; ++i) {
    internal_->attrs[i].second = other.valid_masks()[i];
    if (other.valid_masks()[i]) {
      ++(internal_->size);
      internal_->attrs[i].first = other.attrs()[i];
    }
  }
}

AttrMap::AttrMap(const UserOpConf& user_op_conf) : internal_(std::make_shared<AttrMap::>()) {
  internal_->attr_names.reset(new small_vector<std::string, kInitializedSize>());
  for (const auto& kv : user_op_conf.attr()) {
    auto cpp_attr_value = user_op::AttrValueUtil::ToCppAttrValue(kv.second);
    if (cpp_attr_value.IsOk()) {
      ++(internal_->size);
      internal_->attr_names->emplace_back(kv.first);
      internal_->attrs.emplace_back(CHECK_JUST(cpp_attr_value), true);

      HashCombine(&internal_->hash_value, kv.first.size());
      HashCombine(&internal_->hash_value, internal_->attrs.back().first->hash_value());
    } else {
      LOG(ERROR) << user_op_conf.DebugString()
                 << " failed to convert to cpp attr value, key: " << kv.first;
    }
  }
  internal_->capacity = internal_->size;
}

AttrMap& AttrMap::operator=(const AttrMap& other) {
  internal_ = other.internal_;
  return *this;
}

bool AttrMap::operator==(const AttrMap& other) const {
  if (internal_->size != other.internal_->size
      || internal_->hash_value != other.internal_->hash_value) {
    return false;
  }
  for (int i = 0; i < std::min(internal_->size, other.internal_->size); ++i) {
    if (internal_->attrs[i].second != other.internal_->attrs[i].second) { return false; }
    if (internal_->attrs[i].second) {
      if ((*internal_->attr_names)[i] != (*other.internal_->attr_names)[i]) { return false; }
      if (*(internal_->attrs[i].first) != *(other.internal_->attrs[i].first)) { return false; }
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
  for (int i = 0; i < internal_->capacity; ++i) {
    if (internal_->attrs[i].second && attr_name == (*internal_->attr_names)[i]) {
      return internal_->attrs[i].first;
    }
  }
  static const std::shared_ptr<const user_op::AttrVal> none;
  return none;
}

bool AttrMap::HasAttr4Name(const std::string& attr_name) const {
  return Attr4Name(attr_name) != nullptr;
}

AttrMap::const_iterator::const_iterator(size_t pos, const AttrMap::*internal)
    : pos_(pos), internal_(internal) {
  UpdateKV();
}

AttrMap::const_iterator& AttrMap::const_iterator::operator++() {
  ++pos_;
  UpdateKV();
  return *this;
}

void AttrMap::const_iterator::UpdateKV() {
  while (pos_ < internal_->capacity) {
    if (internal_->attrs[pos_].second) { break; }
    ++pos_;
  }
  if (pos_ < internal_->capacity) {
    kv_.first = (*internal_->attr_names)[pos_];
    kv_.second = internal_->attrs[pos_].first;
  }
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
