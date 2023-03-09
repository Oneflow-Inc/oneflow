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
#ifndef ONEFLOW_CORE_FRAMEWORK_LAYOUT_H_
#define ONEFLOW_CORE_FRAMEWORK_LAYOUT_H_
#include <string>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

enum class LayoutType {
  kStrided,
};

#define LAYOUT_SEQ OF_PP_MAKE_TUPLE_SEQ(Strided)

class Layout final {
 public:
  Layout(const Layout&) = default;
  Layout(Layout&&) = delete;
  explicit Layout(LayoutType layout_type) : layout_type_(layout_type) {}
  ~Layout() = default;

  bool operator==(const Layout& other) const { return this->layout_type() == other.layout_type(); }

  const std::string& name() const;

  LayoutType layout_type() const { return layout_type_; }
  static Symbol<Layout> Get(LayoutType);
#define DECLARE_GET_LAYOUT_TYPE_FUNCTION(layout_type) static Symbol<Layout> layout_type();
  OF_PP_FOR_EACH_TUPLE(DECLARE_GET_LAYOUT_TYPE_FUNCTION, LAYOUT_SEQ)
#undef DECLARE_GET_LAYOUT_TYPE_FUNCTION

 private:
  LayoutType layout_type_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Layout> final {
  size_t operator()(const oneflow::Layout& layout) const {
    return static_cast<size_t>(layout.layout_type());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_LAYOUT_H_
