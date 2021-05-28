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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/function_traits.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename F, typename R, typename T, int nleft, int index>
struct unpack_call_dispatcher {
  template<typename... Args>
  static R apply(const F& f, const std::vector<T>& args, Args&&... unpacked_args) {
    return unpack_call_dispatcher<F, R, T, nleft - 1, index + 1>::apply(
        f, args, std::forward<Args>(unpacked_args)..., args[index]);
  }
};

template<typename F, typename R, typename T, int index>
struct unpack_call_dispatcher<F, R, T, 0, index> {
  template<typename... Args>
  static R apply(const F& f, const std::vector<T>& args, Args&&... unpacked_args) {
    return f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename F, typename R, typename T>
struct unpack_call {
  static R apply(const F& f, const std::vector<T>& args) {
    constexpr size_t nargs = function_traits<F>::nargs;
    CHECK_EQ(nargs, args.size()) << "Requires " << nargs << " arguments, but " << args.size()
                                 << " is given.";
    return unpack_call_dispatcher<F, R, T, nargs, 0>::apply(f, args);
  }
};

#define INSTANCE_MAYBE_UNPACK_CALL(K, return_fn)                                        \
  template<typename F, typename T>                                                      \
  struct unpack_call<F, K, T> {                                                         \
    static constexpr auto return_fn_ = (return_fn);                                     \
    using R = typename function_traits<decltype(return_fn_)>::return_type;              \
    static R apply(const F& f, const std::vector<T>& args) {                            \
      constexpr size_t nargs = function_traits<F>::nargs;                               \
      CHECK_EQ(nargs, args.size())                                                      \
          << "Requires " << nargs << " arguments, but " << args.size() << " is given."; \
      return (return_fn)(unpack_call_dispatcher<F, K, T, nargs, 0>::apply(f, args));    \
    }                                                                                   \
  };

INSTANCE_MAYBE_UNPACK_CALL(Maybe<one::Tensor>,
                           ([](const Maybe<one::Tensor>& t) { return t.GetPtrOrThrow(); }));
INSTANCE_MAYBE_UNPACK_CALL(Maybe<one::TensorTuple>,
                           ([](const Maybe<one::Tensor>& t) { return t.GetPtrOrThrow(); }));

#undef INSTANCE_MAYBE_UNPACK_CALL

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_
