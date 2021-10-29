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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_

#include "oneflow/api/python/functional/python_arg.h"

#include <tuple>
#include <utility>
#include "oneflow/api/python/framework/throw.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/function_traits.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename F, typename R>
struct unpack_call_dispatcher {
  template<size_t... I>
  static R apply(const F& f, const std::vector<PythonArg>& args, std::index_sequence<I...>) {
    return f(args[I]
                 .As<oneflow::detail::remove_cvref_t<typename std::tuple_element<
                     I, typename function_traits<F>::args_type>::type>>()...);
  }
};

template<typename T>
inline py::object CastToPyObject(T&& t) {
  return py::cast(t);
}

template<>
inline py::object CastToPyObject<Maybe<Tensor>>(Maybe<Tensor>&& t) {
  return py::cast(t.GetPtrOrThrow());
}

template<>
inline py::object CastToPyObject<Maybe<TensorTuple>>(Maybe<TensorTuple>&& t) {
  const auto& tensor_tuple = t.GetPtrOrThrow();
  py::tuple tup(tensor_tuple->size());
  for (int i = 0; i < tensor_tuple->size(); ++i) { tup[i] = py::cast(tensor_tuple->at(i)); }
  return py::cast<py::object>(tup);
}

template<>
inline py::object CastToPyObject<Maybe<void>>(Maybe<void>&& t) {
  t.GetOrThrow();
  return py::none();
}

template<typename F>
py::object unpack_call(const F& f, const std::vector<PythonArg>& args) {
  constexpr size_t nargs = function_traits<F>::nargs;
  CHECK_EQ_OR_THROW(nargs, args.size())
      << "Requires " << nargs << " arguments, but " << args.size() << " is given.";
  using R = typename function_traits<F>::return_type;
  return CastToPyObject(
      unpack_call_dispatcher<F, R>::apply(f, args, std::make_index_sequence<nargs>{}));
}

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_
