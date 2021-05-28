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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_FUNCTION_DEF_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_FUNCTION_DEF_H_

#include <memory>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>

#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/core/functional/value_types.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

struct ReturnDef {
  ReturnDef() : type(kINVALID) {}
  ReturnDef(const ValueType& _type) : type(_type) {}
  ValueType type;
};

struct ArgumentDef {
  ArgumentDef() : name(""), type(kINVALID), has_default_value(false) {}
  ArgumentDef(const std::string& _name, const ValueType& _type)
      : name(_name), type(_type), has_default_value(false) {}

  template<typename T>
  ArgumentDef(const std::string& _name, const T& _default_value)
      : name(_name), type(ValueTypeOf<T>()), has_default_value(true) {
    default_value = std::make_shared<detail::ValueImpl<T>>(_default_value);
  }

  std::string name;
  ValueType type;

  bool has_default_value;
  std::shared_ptr<const detail::Value> default_value;
};

struct FunctionDef {
  std::string name;
  ReturnDef return_def;
  std::vector<ArgumentDef> argument_def;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_FUNCTION_DEF_H_
