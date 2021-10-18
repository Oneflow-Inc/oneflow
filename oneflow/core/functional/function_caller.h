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
#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTION_CALLER_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTION_CALLER_H_

#include <functional>
#include <utility>
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace one {
namespace functional {

template<typename T>
struct pointer_inside_maybe {};

template<typename T>
struct pointer_inside_maybe<Maybe<T>> {
  using type = std::shared_ptr<T>;
};

template<typename T>
class Caller;

template<typename R, typename... Args>
class Caller<R(Args...)> {
 public:
  using first_f_type = std::function<R(Args...)>;
  using f_type = std::function<R(const typename pointer_inside_maybe<R>::type&)>;

  explicit Caller(first_f_type&& f) : fn_(std::move(f)) {}

  explicit Caller(const first_f_type& f) : fn_(f) {}

  R call(Args&&... args) const { return fn_(std::forward<Args>(args)...); }

  const Caller<R(Args...)> then(f_type&& f) const {
    auto fn_ = std::move(this->fn_);
    return Caller<R(Args...)>(
        [fn_, f](Args&&... args) -> R { return f(JUST(fn_(std::forward<Args>(args)...))); });
  }

  const Caller<R(Args...)> operator>>(f_type&& f) const { return then(std::forward(f)); }

 private:
  std::function<R(Args...)> fn_;
};

#define make_caller(f) Caller<decltype(f)>(f)

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTION_CALLER_H_
