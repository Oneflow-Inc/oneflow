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
#ifndef ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_

#include "llvm/ADT/StringRef.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

class CachedMutableAttrMap {
 public:
  static constexpr int kInitializedSize = 4;
  static_assert(kInitializedSize > 0, "kInitializedSize must be greater than 0");

  CachedMutableAttrMap()
      : count_(0),
        hash_value_(0),
        attr_names_(
            std::make_shared<small_vector<std::string, kInitializedSize>>(kInitializedSize)) {
    valid_masks_.resize(kInitializedSize);
    attrs_.resize(kInitializedSize);
  }
  ~CachedMutableAttrMap() = default;

  size_t size() const { return count_; }
  size_t hash_value() const { return hash_value_; }

  const std::shared_ptr<small_vector<std::string, kInitializedSize>>& attr_names() const {
    return attr_names_;
  }

  const small_vector<bool, kInitializedSize>& valid_masks() const { return valid_masks_; }
  const small_vector<std::shared_ptr<user_op::AttrVal>, kInitializedSize>& attrs() const {
    return attrs_;
  }

  inline void reset() {
    hash_value_ = 0;
    // mark all cached attributes as illegal values
    memset(valid_masks_.data(), 0, count_);
  }

  template<typename T>
  inline void SetAttr(const char* attr_name, const T& attr_val);

 private:
  inline void InternalEnlarge();

  // the actually count of all attributes
  size_t count_;
  size_t hash_value_;

  struct Hash {
    size_t operator()(const llvm::StringRef& val) const {
      return HashCombine(val.size(), static_cast<size_t>(val.data()[0] - '0'));
    }
  };
  HashMap<llvm::StringRef, int, Hash> name_indices_;

  // `attr_names_` will be shared with other AttrMap
  std::shared_ptr<small_vector<std::string, kInitializedSize>> attr_names_;

  small_vector<bool, kInitializedSize> valid_masks_;
  small_vector<std::shared_ptr<user_op::AttrVal>, kInitializedSize> attrs_;
};

template<typename T>
void CachedMutableAttrMap::SetAttr(const char* attr_name, const T& attr_val) {
  auto it = name_indices_.find(attr_name);
  if (it == name_indices_.end()) {
    if (count_ >= attrs_.size()) { InternalEnlarge(); }
    attrs_[count_] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
    valid_masks_[count_] = true;
    (*attr_names_)[count_] = attr_name;
    it = name_indices_.emplace((*attr_names_)[count_], count_).first;
    ++count_;
  } else {
    if (/*attrs_[i].v->value_type() != user_op::GetAttrType<T>::value ||*/
        *static_cast<const T*>(attrs_[it->second]->Ptr()) != attr_val) {
      attrs_[it->second] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
    }
    valid_masks_[it->second] = true;
  }
  HashCombine(&hash_value_, (*attr_names_)[it->second].size());
  HashCombine(&hash_value_, std::hash<T>()(attr_val));
}

void CachedMutableAttrMap::InternalEnlarge() {
  size_t capacity = count_ * 2;
  valid_masks_.resize(capacity);
  attrs_.resize(capacity);
  // expand `attr_names_`
  auto attr_names = std::make_shared<small_vector<std::string, kInitializedSize>>();
  attr_names->resize(capacity);
  std::copy(attr_names_->begin(), attr_names_->end(), attr_names->begin());
  attr_names_ = std::move(attr_names);
}

#define THREAD_LOCAL_MUT_ATTR_MAP                   \
  []() {                                            \
    thread_local static CachedMutableAttrMap attrs; \
    attrs.reset();                                  \
    return &attrs;                                  \
  }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_
