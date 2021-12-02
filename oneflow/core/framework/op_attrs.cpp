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
#include "oneflow/core/framework/op_attrs.h"
#include "oneflow/core/framework/op_interp_ctx.h"

namespace oneflow {

size_t OpAttrs::count(const std::string& attr_name) const {
  return ctx_->AttrNamesSet().count(attr_name);
}

Maybe<AttrVal> OpAttrs::at(const std::string& attr_name) const { return ctx_->GetAttr(attr_name); }
Maybe<AttrVal> OpAttrs::operator[](const std::string& attr_name) const {
  return ctx_->GetAttr(attr_name);
}

OpAttrs::const_iterator OpAttrs::begin() const {
  const auto& attrs = ctx_->AttrNamesSet();
  return const_iterator(attrs.cbegin(), attrs.cend(), this);
}
OpAttrs::const_iterator OpAttrs::end() const {
  const auto& attrs = ctx_->AttrNamesSet();
  return const_iterator(attrs.cend(), attrs.cend(), this);
}

}  // namespace oneflow
