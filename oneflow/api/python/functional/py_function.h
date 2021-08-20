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
#include <pybind11/pybind11.h>

#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/unpack_call.h"
#include "oneflow/api/python/framework/throw.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

// The argument parsing refers to the implementation of Pytorch.
template<typename SchemaT>
inline bool ParseArgs(const py::args& args, const py::kwargs& kwargs,
                      std::vector<PythonArg>* parsed_args, bool raise_exception) {
  const auto& function = SchemaT::function_def;
  bool treat_args_as_intlist = false;
  size_t nargs = args.size();
  size_t remaining_kwargs = kwargs.size();

  if (SchemaT::max_pos_args == 1) {
    const auto& type = function.argument_def.at(0).type;
    treat_args_as_intlist = (type == kINT32_LIST || type == kUINT32_LIST || type == kINT64_LIST
                             || type == kUINT64_LIST);
  }
  if (nargs > SchemaT::max_pos_args && !treat_args_as_intlist) {
    if (raise_exception) {
      THROW(TypeError) << function.name << "(): takes " << SchemaT::max_pos_args
                       << " positional arguments but " << nargs << " were given.";
    }
    return false;
  }
  int arg_pos = 0;
  for (int i = 0; i < function.argument_def.size(); ++i) {
    const auto& param = function.argument_def.at(i);
    py::object obj;
    if (arg_pos == 0 && treat_args_as_intlist && !param.keyword_only) {
      obj = args;
      arg_pos = nargs;
    } else if (arg_pos < nargs) {
      if (param.keyword_only) {
        if (raise_exception) {
          THROW(TypeError) << function.name << "(): argument '" << param.name
                           << "' is keyword only.";
        }
        return false;
      }
      obj = args[arg_pos++];
    } else {
      if (kwargs.contains(param.name.c_str())) {
        obj = kwargs[param.name.c_str()];
        remaining_kwargs--;
      }
    }

    if (!obj && !param.has_default_value) {
      if (raise_exception) {
        THROW(TypeError) << function.name << "(): missing required argument " << param.name;
      }
      return false;
    } else if (obj) {
      PythonArg arg(obj);
      if (!PythonArgCheck(arg, param.type)) {
        if (raise_exception) {
          THROW(TypeError) << function.name << "(): argument '" << param.name << "' must be "
                           << ValueTypeName(param.type).GetOrThrow() << ", not "
                           << Py_TYPE(obj.ptr())->tp_name;
        }
        return false;
      }
      parsed_args->at(i) = std::move(arg);
    } else {
      parsed_args->at(i) = PythonArg(param.default_value);
    }
  }
  if (remaining_kwargs > 0) {
    if (raise_exception) { THROW(TypeError); }
    return false;
  }
  return true;
}

template<size_t Size, typename SchemaT, typename... SchemaListT>
struct PyFunctionImpl {
  inline static py::object apply(const py::args& args, const py::kwargs& kwargs,
                                 bool raise_exception) {
    std::vector<PythonArg> parsed_args(SchemaT::max_args);
    if (ParseArgs<SchemaT>(args, kwargs, &parsed_args, raise_exception)) {
      return detail::unpack_call(*SchemaT::func, parsed_args);
    }
    return PyFunctionImpl<Size - 1, SchemaListT...>::apply(args, kwargs, raise_exception);
  }
};

template<typename SchemaT>
struct PyFunctionImpl<1, SchemaT> {
  inline static py::object apply(const py::args& args, const py::kwargs& kwargs,
                                 bool raise_exception) {
    std::vector<PythonArg> parsed_args(SchemaT::max_args);
    if (!ParseArgs<SchemaT>(args, kwargs, &parsed_args, raise_exception)) {
      THROW(TypeError) << SchemaT::function_def.name << "(): no matching function to call.";
    }
    return detail::unpack_call(*SchemaT::func, parsed_args);
  }
};

}  // namespace detail

template<typename... SchemaListT>
inline py::object PyFunction(const py::args& args, const py::kwargs& kwargs) {
  static constexpr size_t schema_size = sizeof...(SchemaListT);
  return detail::PyFunctionImpl<schema_size, SchemaListT...>::apply(
      args, kwargs, /*raise_exception=*/schema_size == 1);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
