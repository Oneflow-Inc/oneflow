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
  ReturnDef(const ValueType& t) : type(t) {}
  ValueType type;
};

struct ArgumentDef {
  ArgumentDef(const std::string& arg_name, const ValueType& arg_type, bool arg_keyword_only = false,
              bool arg_optional = false)
      : name(arg_name),
        type(arg_type),
        keyword_only(arg_keyword_only),
        optional(arg_optional),
        has_default_value(false) {}

  template<typename T>
  ArgumentDef(const std::string& arg_name, const T& arg_val, bool arg_keyword_only = false,
              bool arg_optional = false)
      : name(arg_name),
        type(ValueTypeOf<T>()),
        keyword_only(arg_keyword_only),
        optional(arg_optional),
        has_default_value(true) {
    default_value = std::make_shared<detail::TypedImmediate<T>>(arg_val);
  }

  std::string name;
  ValueType type;

  bool keyword_only;
  bool optional;
  bool has_default_value;
  std::shared_ptr<const detail::Immediate> default_value;
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
