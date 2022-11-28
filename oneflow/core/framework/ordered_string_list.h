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
#ifndef ONEFLOW_CORE_FRAMEWORK_ORDERED_STRING_LIST_H_
#define ONEFLOW_CORE_FRAMEWORK_ORDERED_STRING_LIST_H_

#include "llvm/ADT/StringRef.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/small_vector.h"

namespace oneflow {

template<int N>
class OrderedStringList {
 public:
  OrderedStringList() = default;

  size_t size() const { return strings_.size(); }

  void emplace_back(llvm::StringRef s) {
    strings_.emplace_back(std::make_shared<std::string>(s.str()));
    order_.emplace(*strings_.back(), order_.size());
  }

  int order(llvm::StringRef s) const {
    const auto& it = order_.find(s);
    if (it == order_.end()) { return -1; }
    return it->second;
  }

  const std::string& operator[](int idx) { return *(strings_[idx]); }

 private:
  struct Hash {
    size_t operator()(llvm::StringRef val) const {
      return HashCombine(val.size(), val.size() > 0 ? static_cast<size_t>(val.data()[0] - '0') : 0);
    }
  };
  HashMap<llvm::StringRef, int, Hash> order_;
  // Use shared_ptr to prevent the appended element from being freed when the
  // vector increases
  small_vector<std::shared_ptr<std::string>, N> strings_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ORDERED_STRING_LIST_H_
