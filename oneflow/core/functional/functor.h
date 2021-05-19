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

#include "oneflow/core/functional/value_types.h"
#include "oneflow/core/functional/function_signature.h"
#include "oneflow/core/functional/function_traits.h"

namespace oneflow {
namespace one {
namespace functional {

class FunctionBody {
 public:
  FunctionBody() = default;
  virtual ~FunctionBody() = default;
  virtual operator void*() = 0;
};

template<typename T>
class FunctionBodyImpl;

template<typename R, typename... Args>
class FunctionBodyImpl<R(Args...)> : public FunctionBody {
 public:
  template<typename Func>
  FunctionBodyImpl(Func func);

  operator void*() override { return &func_; }

 private:
  std::function<R(Args...)> func_;
};

template<typename R, typename... Args>
template<typename Func>
FunctionBodyImpl<R(Args...)>::FunctionBodyImpl(Func func) {
  using FuncType = typename detail::function_traits<Func>::func_type;
  static_assert(std::is_same<FuncType, R(Args...)>::value);
  func_ = [func](Args... args) { return func(std::forward<Args>(args)...); };
}

class Functor {
 public:
  Functor(const std::shared_ptr<FunctionBody>& body, const FunctionSignature& signatrue)
      : body_(body), signatrue_(signatrue) {}

  template<typename R, typename... Args>
  R call(Args... args) const {
    if (!detail::CheckFunctionSignature<R(Args...)>(signatrue_).Ok()) {
      LOG(FATAL) << "The function was called with wrong arguments.";
    }
    using FuncType = std::function<R(Args...)>;
    auto* func = reinterpret_cast<FuncType*>(body_->operator void*());
    return func->operator()(std::forward<Args>(args)...);
  }

 private:
  std::shared_ptr<FunctionBody> body_;
  FunctionSignature signatrue_;
};

class PackedFunctor {
 public:
  virtual ~PackedFunctor() = default;

  template<typename Func>
  static PackedFunctor MakePackedFunctor(const std::string& func_name, Func func);

  template<typename R, typename... Args>
  R call(Args... args) const {
    return functor_.call<R, Args...>(std::forward<Args>(args)...);
  }

 private:
  PackedFunctor(const std::string& func_name, const Functor& functor)
      : func_name_(func_name), functor_(functor) {}

  std::string func_name_;
  Functor functor_;
};

template<typename Func>
/*static*/ PackedFunctor PackedFunctor::MakePackedFunctor(const std::string& func_name, Func func) {
  // static_assert(is_callable(func));
  using func_type = typename detail::function_traits<Func>::func_type;
  auto body = std::make_shared<FunctionBodyImpl<func_type>>(func);
  FunctionSignature signatute = detail::PackFunctionSignature<func_type>::pack();
  Functor functor(body, signatute);
  return PackedFunctor(func_name, std::move(functor));
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_H_
