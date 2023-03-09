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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Symbol<Layout> Layout::Get(LayoutType layout_type) {
  static const HashMap<LayoutType, Symbol<Layout>> layout_type2layout{
#define MAKE_ENTRY(layout_type) {OF_PP_CAT(LayoutType::k, layout_type), layout_type()},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, LAYOUT_SEQ)
#undef MAKE_ENTRY
  };
  return layout_type2layout.at(layout_type);
}

const std::string& GetLayoutTypeName(LayoutType layout_type) {
  static const HashMap<LayoutType, std::string> layout_type2name{
      {LayoutType::kStrided, "oneflow.strided"}};
  return layout_type2name.at(layout_type);
};

const std::string& Layout::name() const { return GetLayoutTypeName(layout_type_); }

#define DEFINE_GET_LAYOUT_TYPE_FUNCTION(layout_type)                                     \
  Symbol<Layout> Layout::layout_type() {                                                 \
    static const auto& layout = SymbolOf(Layout(OF_PP_CAT(LayoutType::k, layout_type))); \
    return layout;                                                                       \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_LAYOUT_TYPE_FUNCTION, LAYOUT_SEQ)
#undef DEFINE_GET_LAYOUT_TYPE_FUNCTION

}  // namespace oneflow
