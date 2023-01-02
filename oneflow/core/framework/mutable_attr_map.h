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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/ordered_string_list.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class MutableAttrMap {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MutableAttrMap);

  explicit MutableAttrMap(const std::vector<std::string>& attr_names)
      : max_size_(attr_names.size()),
        valid_masks_(max_size_, 0),
        ordered_attr_names_(std::make_shared<OrderedStringList<8>>()) {
    for (const auto& attr_name : attr_names) { ordered_attr_names_->emplace_back(attr_name); }
    attrs_.resize(max_size_);
  }

  ~MutableAttrMap() = default;

  size_t max_size() const { return max_size_; }

  const std::shared_ptr<OrderedStringList<8>>& ordered_attr_names() const {
    return ordered_attr_names_;
  }
  const small_vector<bool, 8>& valid_masks() const { return valid_masks_; }
  const small_vector<std::shared_ptr<user_op::AttrVal>, 8>& attrs() const { return attrs_; }

  inline void reset() {
    // mark all cached attributes as illegal values
    memset(valid_masks_.data(), 0, max_size_);
  }

  template<typename T>
  inline void SetAttr(const char* attr_name, const T& attr_val) {
    auto idx = ordered_attr_names_->order(attr_name);
    CHECK_OR_THROW(idx != -1) << "has no attribute named " << attr_name;
    SetAttrNoThrow(idx, attr_val);
  }

  template<int I, typename T>
  inline void SetAttr(const T& attr_val) {
    CHECK_LT_OR_THROW(I, max_size_)
        << "index " << I << " is out of bound, and the max size is " << max_size_;
    SetAttrNoThrow(I, attr_val);
  }

  template<typename... Args>
  inline void SetAllAttrs(Args&&... args) {
    CHECK_EQ_OR_THROW(sizeof...(args), max_size_)
        << "requires " << max_size_ << " arguments, but gives " << sizeof...(args);
    SetAttrNoThrow<Args...>(std::forward<Args>(args)...,
                            std::make_index_sequence<sizeof...(args)>{});
  }

 private:
  template<typename T, typename std::enable_if<!std::is_same<T, NullOptType>::value
                                                   && !internal::IsOptional<T>::value,
                                               int>::type = 0>
  inline void SetAttrNoThrow(int idx, const T& attr_val) {
    valid_masks_[idx] = true;
    if (!attrs_[idx] /*|| attrs_[idx]->type() != user_op::GetAttrType<T>::value*/
        || *static_cast<const T*>(attrs_[idx]->Ptr()) != attr_val) {
      attrs_[idx] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
    }
  }

  template<typename T, typename std::enable_if<internal::IsOptional<T>::value, int>::type = 0>
  inline void SetAttrNoThrow(int idx, const T& attr_val) {
    if (attr_val) {
      using U = typename T::value_type;
      SetAttrNoThrow(idx, attr_val.value_or(U()));
    }
  }

  template<typename T, typename std::enable_if<std::is_same<T, NullOptType>::value, int>::type = 0>
  inline void SetAttrNoThrow(int idx, const T&) {}

  template<typename... Args, size_t... I>
  inline void SetAttrNoThrow(Args&&... args, std::index_sequence<I...>) {
    __attribute__((__unused__)) int dummy[] = {(SetAttrNoThrow(I, std::forward<Args>(args)), 0)...};
  }

  // The actually count of all attributes
  size_t max_size_;
  small_vector<bool, 8> valid_masks_;
  small_vector<std::shared_ptr<user_op::AttrVal>, 8> attrs_;
  // The ordered attribute names is determined and should be shared
  // between other AttrMap
  std::shared_ptr<OrderedStringList<8>> ordered_attr_names_;
};

#define THREAD_CACHED_MUTABLE_ATTR_MAP(...)                                          \
  []() -> MutableAttrMap& {                                                          \
    thread_local static MutableAttrMap attrs(std::vector<std::string>{__VA_ARGS__}); \
    attrs.reset();                                                                   \
    return attrs;                                                                    \
  }()

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_
