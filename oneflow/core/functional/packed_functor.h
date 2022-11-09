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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_

#include <memory>

#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/type_traits.h"

namespace oneflow {
namespace one {
namespace functional {

template<typename T>
using remove_cvref_t = oneflow::detail::remove_cvref_t<T>;

template<typename T>
class PackedFunctor;

template<typename R, typename... Args>
class PackedFunctor<R(Args...)> {
 public:
  PackedFunctor(const std::string& func_name, const std::function<R(Args...)>& impl)
      : func_name_(func_name), impl_(impl) {}

  virtual ~PackedFunctor() = default;

  template<typename... TArgs>
  R call(TArgs&&... args) const {
    return impl_(std::forward<TArgs>(args)...);
  }

 private:
  std::string func_name_;
  std::function<R(Args...)> impl_;
};

template<typename T>
class PackedFunctorMaker;

template<typename R, typename... Args>
class PackedFunctorMaker<R(Args...)> {
 public:
  using FType = R(const remove_cvref_t<Args>&...);

  template<typename Func,
           typename std::enable_if<
               std::is_same<typename function_traits<Func>::func_type, R(Args...)>::value,
               int>::type = 0>
  static PackedFunctor<FType> make(const std::string& func_name, const Func& func) {
    return PackedFunctor<FType>(func_name, [func](const remove_cvref_t<Args>&... args) -> R {
      return func(std::forward<const remove_cvref_t<Args>&>(args)...);
    });
  }
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_
