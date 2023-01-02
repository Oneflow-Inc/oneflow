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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_DEFINITION_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_DEFINITION_H_
#include <string>

#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace user_op {
class AttrVal;
}  // namespace user_op
using AttrVal = user_op::AttrVal;

class OpDefinitionBase {
 public:
  virtual ~OpDefinitionBase() = default;
  virtual Maybe<AttrVal> Attr(const std::string& attr_name) const = 0;
  virtual const HashSet<std::string>& AttributeNames() const = 0;

 protected:
  OpDefinitionBase() = default;
};

template<typename Derived>
class OpDefinition : public OpDefinitionBase {
 public:
  virtual ~OpDefinition() = default;
  const HashSet<std::string>& AttributeNames() const override { return Derived::AttrNames(); }

 protected:
  OpDefinition() : OpDefinitionBase() {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_DEFINITION_H_
