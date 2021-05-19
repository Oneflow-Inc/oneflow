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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTION_SIGNATURE_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTION_SIGNATURE_H_

#include <memory>
#include <tuple>

#include "oneflow/core/functional/function_traits.h"
#include "oneflow/core/functional/value_types.h"

namespace oneflow {
namespace one {
namespace functional {

struct FunctionSignature {
  ValueType return_type;
  std::vector<ValueType> argument_types;
};

namespace detail {

template<typename T>
inline ValueType PackType() {
  return ValueTypeOf<typename std::decay<T>::type>();
}

template<typename... Args>
struct PackTypeListImpl;

template<>
struct PackTypeListImpl<> {
  static void pack(std::vector<ValueType>* packed_types) {}
};

template<typename T, typename... Args>
struct PackTypeListImpl<T, Args...> {
  static void pack(std::vector<ValueType>* packed_types) {
    packed_types->emplace_back(PackType<T>());
    PackTypeListImpl<Args...>::pack(packed_types);
  }
};

template<typename... Args>
inline std::vector<ValueType> PackTypeList() {
  std::vector<ValueType> packed_types;
  detail::PackTypeListImpl<Args...>::pack(&packed_types);
  return packed_types;
}

template<typename R, typename... Args>
inline FunctionSignature PackFunctionSignatureImpl() {
  FunctionSignature signature;
  signature.return_type = PackType<R>();
  signature.argument_types = detail::PackTypeList<Args...>();
  return signature;
}

template<typename T>
class PackFunctionSignature;

template<typename R, typename... Args>
class PackFunctionSignature<R(Args...)> {
 public:
  static FunctionSignature pack() {
    static auto signature = PackFunctionSignatureImpl<R, Args...>();
    return signature;
  }
};

template<typename T>
class CheckFunctionSignature;

template<typename R, typename... Args>
class CheckFunctionSignature<R(Args...)> {
 public:
  CheckFunctionSignature(const FunctionSignature& signature) {
    status_ = CheckFunctionSignatureImpl(signature);
  }

  bool Ok() { return status_; }

 private:
  bool CheckFunctionSignatureImpl(const FunctionSignature& signature);
  bool status_;
};

template<typename R, typename... Args>
bool CheckFunctionSignature<R(Args...)>::CheckFunctionSignatureImpl(
    const FunctionSignature& signature) {
  static ValueType return_type = detail::PackType<R>();
  if (signature.return_type != return_type) { return false; }
  static std::vector<ValueType> argument_types = detail::PackTypeList<Args...>();
  if (argument_types != signature.argument_types) { return false; }
  return true;
}

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTION_SIGNATURE_H_
