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
#include "oneflow/api/python/functional/py_function.h"
#include "oneflow/api/python/functional/common.h"

namespace oneflow {
namespace one {
namespace functional {

void ReportKwargsError(const py::kwargs& kwargs, const FunctionDef& function, size_t max_pos_args) {
  for (auto it = kwargs.begin(); it != kwargs.end(); ++it) {
    if (!PyStringCheck(it->first.ptr())) {
      THROW(TypeError) << function.name << "(): keywords must be strings.";
    }
    bool unexpected_param = true;
    const std::string key = PyStringAsString(it->first.ptr()).GetOrThrow();
    for (const auto& arg : function.argument_def) {
      if (arg.name == key) {
        unexpected_param = false;
        break;
      }
    }
    if (unexpected_param) {
      THROW(TypeError) << function.name  // NOLINT
                       << "(): got an unexpected keyword argument '" << key << "'";
    } else {
      THROW(TypeError) << function.name  // NOLINT
                       << "(): got multiple values for argument '" << key << "'";
    }
  }
  THROW(TypeError) << function.name << "(): kwargs unknown error.";
}

// The argument parsing refers to the implementation of Pytorch.
bool ParseArgs(const py::args& args, const py::kwargs& kwargs, std::vector<PythonArg>* parsed_args,
               const FunctionDef& function, size_t max_pos_args, bool raise_exception) {
  bool treat_args_as_list = false;
  size_t nargs = args.size();
  size_t remaining_kwargs = kwargs.size();

  if (max_pos_args == 1) {
    const auto& type = function.argument_def.at(0).type;
    treat_args_as_list = IsIntegralListType(type) || type == kSHAPE || type == kTENSOR_TUPLE;
  }
  if (nargs > max_pos_args && !treat_args_as_list) {
    if (raise_exception) {
      THROW(TypeError) << function.name << "(): takes " << max_pos_args
                       << " positional arguments but " << nargs << " were given.";
    }
    return false;
  }
  int arg_pos = 0;
  for (int i = 0; i < function.argument_def.size(); ++i) {
    const auto& param = function.argument_def.at(i);
    py::object obj;
    if (arg_pos < nargs) {
      if (param.keyword_only) {
        if (raise_exception) {
          THROW(TypeError) << function.name << "(): argument '" << param.name
                           << "' is keyword only.";
        }
        return false;
      }
      obj = args[arg_pos];
    } else {
      if (kwargs.contains(param.name.c_str())) {
        obj = kwargs[param.name.c_str()];
        remaining_kwargs--;
      }
    }

    if (obj) {
      if (arg_pos == 0 && treat_args_as_list && !param.keyword_only
          && (PyLong_Check(obj.ptr()) || PyTensorCheck(obj.ptr()))) {
        obj = args;
        arg_pos = nargs;
      } else {
        arg_pos++;
      }
      PythonArg arg(obj, param.size);
      if ((obj == Py_None && param.optional) || PythonArgCheck(arg, param.type)) {
        parsed_args->at(i) = std::move(arg);
      } else {
        if (raise_exception) {
          THROW(TypeError) << function.name << "(): argument '" << param.name << "' must be "
                           << ValueTypeName(param.type).GetOrThrow() << ", not "
                           << Py_TYPE(obj.ptr())->tp_name;
        }
        return false;
      }
    } else {
      if (!param.has_default_value) {
        if (raise_exception) {
          THROW(TypeError) << function.name << "(): missing required argument " << param.name;
        }
        return false;
      }
      parsed_args->at(i) = PythonArg(param.default_value);
    }
  }
  if (remaining_kwargs > 0) {
    if (raise_exception) { ReportKwargsError(kwargs, function, max_pos_args); }
    return false;
  }
  return true;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
