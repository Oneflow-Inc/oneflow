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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_SCHEMA_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_SCHEMA_H_

#include <string>
#include <vector>

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/attr_value.h"

namespace oneflow {
namespace one {

class OpSchema {
 public:
  template<typename T>
  Maybe<const T&> GetAttr(const char* attr_name) const {
    return *reinterpret_cast<const T*>(JUST(GetAttr(attr_name)));
  }

 protected:
  virtual Maybe<const void*> GetAttr(const char* attr_name) const = 0;
};

class ConstantOpSchema : public OpSchema {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "shape")) {
      return (const void*)&shape;
    } else if (!strcmp(attr_name, "dtype")) {
      return (const void*)&dtype;
    } else if (!strcmp(attr_name, "is_floating_value")) {
      return (const void*)&is_floating_value;
    } else if (!strcmp(attr_name, "integer_value")) {
      return (const void*)&integer_value;
    } else if (!strcmp(attr_name, "floating_value")) {
      return (const void*)&floating_value;
    } else if (!strcmp(attr_name, "nd_sbp")) {
      return (const void*)&nd_sbp;
    } else {
      return Error::RuntimeError() << "Op schema has no attribute named " << attr_name;
    }
  }

 public:
  Shape shape;
  DataType dtype;
  bool is_floating_value;
  int64_t integer_value;
  double floating_value;
  std::vector<std::string> nd_sbp;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_SCHEMA_H_
