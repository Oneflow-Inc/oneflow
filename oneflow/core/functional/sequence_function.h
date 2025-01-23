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
#ifndef ONEFLOW_CORE_FUNCTIONAL_SEQUENCE_FUNCTION_H_
#define ONEFLOW_CORE_FUNCTIONAL_SEQUENCE_FUNCTION_H_

#include <functional>
#include <utility>
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace one {
namespace functional {

template<typename T>
class SequenceFunction;

template<typename R, typename... Args>
class SequenceFunction<R(Args...)> {
 public:
  using f_type = std::function<R(Args...)>;

  explicit SequenceFunction(f_type&& f) : fn_(std::forward<f_type>(f)) {}

  explicit SequenceFunction(const f_type& f) : fn_(f) {}

  template<typename F>
  SequenceFunction<R(Args...)>& then(F&& f) {
    auto fn_ = std::move(this->fn_);
    this->fn_ = [fn_, f](Args&&... args) -> R { return f(JUST(fn_(std::forward<Args>(args)...))); };
    return *this;
  }

  template<typename F>
  SequenceFunction<R(Args...)>& then_if(bool condition, F&& f) {
    return condition ? then(std::forward<F>(f)) : *this;
  }

  template<typename F>
  SequenceFunction<R(Args...)>& operator|(F&& f) {
    return then(std::forward<F>(f));
  }

  R call(Args&&... args) const { return fn_(std::forward<Args>(args)...); }

 private:
  f_type fn_;
};

#define sequence_function(f) SequenceFunction<decltype(f)>(f)

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_SEQUENCE_FUNCTION_H_
