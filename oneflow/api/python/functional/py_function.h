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

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

template<typename SchemaT>
inline py::object PyFunction(py::args args, py::kwargs kwargs) {
  // TODO(): Support multiple function signatures.
  using FType = typename SchemaT::FType;
  using R = typename SchemaT::R;
  CHECK_LE(args.size(), SchemaT::max_positionals)
      << "The maximum count of positional arguments is " << SchemaT::max_positionals;
  CHECK_LE(kwargs.size(), SchemaT::max_keywords)
      << "The maximum count of keyword arguments is " << SchemaT::max_keywords;

  std::vector<PythonArg> _args(SchemaT::max_args);
  for (int i = 0; i < args.size(); ++i) { _args[i] = PythonArg(args[i]); }
  for (int i = args.size(); i < SchemaT::max_args; ++i) {
    const auto& arg = SchemaT::argument_def.at(i);
    if (kwargs.contains(arg.name.c_str())) {
      _args[i] = PythonArg(kwargs[arg.name.c_str()]);
    } else {
      CHECK(arg.has_default_value) << "Argument " << arg.name << " is required";
      _args[i] = PythonArg(arg.default_value);
    }
  }
  return py::cast(detail::unpack_call<FType, R, PythonArg>::apply(*SchemaT::func, _args));
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
