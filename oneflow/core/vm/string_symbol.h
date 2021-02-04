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
#ifndef ONEFLOW_CORE_VM_STRING_DESC_H_
#define ONEFLOW_CORE_VM_STRING_DESC_H_

#include <string>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class StringSymbol final {
 public:
  StringSymbol(const StringSymbol&) = delete;
  StringSymbol(StringSymbol&&) = delete;
  StringSymbol(int64_t symbol_id, const std::string& data);

  ~StringSymbol() = default;

  const Maybe<int64_t>& symbol_id() const { return symbol_id_; }
  const std::string& data() const { return data_; }

 private:
  Maybe<int64_t> symbol_id_;
  std::string data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STRING_DESC_H_
