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
  using first_f_type = std::function<R(Args...)>;
  using f_type = std::function<R(
      const decltype(std::declval<R>().Data_YouAreNotAllowedToCallThisFuncOutsideThisFile())&)>;

  explicit SequenceFunction(first_f_type&& f) : fn_(std::forward<first_f_type>(f)) {}

  explicit SequenceFunction(const first_f_type& f) : fn_(f) {}

  SequenceFunction<R(Args...)>& then(f_type&& f) {
    auto fn_ = std::move(this->fn_);
    this->fn_ = [fn_, f](Args&&... args) -> R { return f(JUST(fn_(std::forward<Args>(args)...))); };
    return *this;
  }

  SequenceFunction<R(Args...)>& then_if(bool condition, f_type&& f) {
    return condition ? then(std::forward<f_type>(f)) : *this;
  }

  SequenceFunction<R(Args...)>& operator<<(f_type&& f) { return then(std::forward<f_type>(f)); }

  R call(Args&&... args) const { return fn_(std::forward<Args>(args)...); }

 private:
  std::function<R(Args...)> fn_;
};

#define sequence_function(f) SequenceFunction<decltype(f)>(f)

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_SEQUENCE_FUNCTION_H_
