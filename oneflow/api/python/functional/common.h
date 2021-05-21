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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_

#include <pybind11/pybind11.h>

#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/function_traits.h"

namespace py = pybind11;

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename T>
inline bool isinstance(py::object obj) {
  return py::isinstance<T>(obj);
}

template<typename R, int nleft, int index, typename Func>
struct unpack_call_dispatcher {
  template<typename... Args>
  static R apply(const Func& f, py::args args, Args&&... unpacked_args) {
    return unpack_call_dispatcher<R, nleft - 1, index + 1, Func>::apply(
        f, args, std::forward<Args>(unpacked_args)..., PythonArg(args[index]));
  }
};

template<typename R, int index, typename Func>
struct unpack_call_dispatcher<R, 0, index, Func> {
  template<typename... Args>
  static R apply(const Func& f, py::args args, Args&&... unpacked_args) {
    return f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename Func, typename R>
struct unpack_call {
  static R apply(const Func& f, py::args args) {
    constexpr size_t nargs = function_traits<Func>::nargs;
    CHECK_EQ(nargs, args.size()) << "Requires " << nargs << " arguments, but " << args.size()
                                 << " is given.";
    return unpack_call_dispatcher<R, nargs, 0, Func>::apply(f, args);
  }
};

#define SPEC_MAYBE_UNPACK_CALL(T, return_fn)                                            \
  template<typename Func>                                                               \
  struct unpack_call<Func, T> {                                                         \
    static constexpr auto return_fn_ = (return_fn);                                     \
    using R = typename function_traits<decltype(return_fn_)>::return_type;              \
    static R apply(const Func& f, py::args args) {                                      \
      constexpr size_t nargs = function_traits<Func>::nargs;                            \
      CHECK_EQ(nargs, args.size())                                                      \
          << "Requires " << nargs << " arguments, but " << args.size() << " is given."; \
      return (return_fn)(unpack_call_dispatcher<T, nargs, 0, Func>::apply(f, args));    \
    }                                                                                   \
  };

SPEC_MAYBE_UNPACK_CALL(Maybe<one::Tensor>,
                       ([](const Maybe<one::Tensor>& t) { return t.GetPtrOrThrow(); }));
SPEC_MAYBE_UNPACK_CALL(Maybe<one::TensorTuple>,
                       ([](const Maybe<one::Tensor>& t) { return t.GetPtrOrThrow(); }));

#undef SPEC_MAYBE_UNPACK_CALL

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_COMMON_H_
