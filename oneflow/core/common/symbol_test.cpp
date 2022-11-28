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
#include "gtest/gtest.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

namespace detail {

class SymObject {
 public:
  SymObject(const std::string& name) : name_(name) {}

  const std::string& name() const { return name_; }

  bool operator==(const SymObject& other) const { return name_ == other.name_; }

 private:
  std::string name_;
};

}  // namespace detail

TEST(Symbol, shared_from_symbol) {
  Symbol<detail::SymObject> symbol(detail::SymObject("SymbolObjectFoo"));
  ASSERT_TRUE(symbol.shared_from_symbol().get()
              == SymbolOf(detail::SymObject("SymbolObjectFoo")).shared_from_symbol().get());
}

}  // namespace test
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::test::detail::SymObject> final {
  size_t operator()(const oneflow::test::detail::SymObject& sym_object) const {
    return std::hash<std::string>()(sym_object.name());
  }
};

}  // namespace std
