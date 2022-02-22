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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_BASE_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_BASE_H_

#include <string>

#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace user_op {
class AttrVal;
}  // namespace user_op
using AttrVal = user_op::AttrVal;

class OpBase {
 public:
  virtual ~OpBase() = default;

  virtual Maybe<AttrVal> GetAttr(const std::string& attr_name) const = 0;

  virtual const HashSet<std::string>& AttrNames() const {
    static const HashSet<std::string> attr_names;
    return attr_names;
  }

 protected:
  OpBase() = default;
};

class FakeOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "`FakeOp` has no attribute.";
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_BASE_H_
