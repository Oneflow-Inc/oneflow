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

#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/unpack_call.h"
#include "oneflow/api/python/framework/throw.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

bool ParseArgs(const py::args& args, const py::kwargs& kwargs,  // NOLINT
               std::vector<PythonArg>* parsed_args, const FunctionDef& function,
               size_t max_pos_args, bool raise_exception);

template<typename... SchemaT>
class PyFunctionDispatcher {
 public:
  static_assert(sizeof...(SchemaT) >= 1, "Requires 1 template argument at least.");
  using T0 = typename std::tuple_element<0, std::tuple<SchemaT...>>::type;

  PyFunctionDispatcher() : schema_size_(sizeof...(SchemaT)), func_name_(T0::function_def.name) {
    signatures_.resize(schema_size_);
    RecursiveInit(std::make_index_sequence<sizeof...(SchemaT)>{});
  }

  template<size_t I0, size_t... I>
  py::object call(const py::args& args, const py::kwargs& kwargs,
                  std::index_sequence<I0, I...>) const {
    using T = typename std::tuple_element<I0, std::tuple<SchemaT...>>::type;
    std::vector<PythonArg> parsed_args(T::max_args);
    if (ParseArgs(args, kwargs, &parsed_args, T::function_def, T::max_pos_args,
                  /*raise_exception*/ schema_size_ == 1)) {
      return detail::unpack_call(*T::func, parsed_args);
    }
    return call(args, kwargs, std::index_sequence<I...>{});
  }

  py::object call(const py::args& args, const py::kwargs& kwargs, std::index_sequence<>) const {
    std::ostringstream ss;
    ss << func_name_
       << "(): received an invalid combination of arguments. The valid signatures are:";
    for (int i = 0; i < signatures_.size(); ++i) {
      ss << "\n\t*" << i << ": " << signatures_.at(i);
    }
    THROW(TypeError) << ss.str();
    return py::none();
  }

 private:
  void RecursiveInit(std::index_sequence<>) {}

  template<size_t I0, size_t... I>
  void RecursiveInit(std::index_sequence<I0, I...>) {
    using T = typename std::tuple_element<I0, std::tuple<SchemaT...>>::type;
    signatures_[I0] = T::signature;
    RecursiveInit(std::index_sequence<I...>{});
  }

 private:
  size_t schema_size_;
  const std::string func_name_;
  std::vector<const char*> signatures_;
};

template<typename... SchemaT>
inline py::object PyFunction(const py::args& args, const py::kwargs& kwargs) {
  static PyFunctionDispatcher<SchemaT...> dispatcher;
  return dispatcher.call(args, kwargs, std::make_index_sequence<sizeof...(SchemaT)>{});
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
