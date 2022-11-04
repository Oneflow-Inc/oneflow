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

// Generated from oneflow/core/functional/functional_api.yaml. DO NOT EDIT!

#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> Add(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("Add"));
  return __op->call(input, other, alpha, inplace);
}

Maybe<one::Tensor> ScalarAdd(const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const Scalar&, bool>("ScalarAdd"));
  return __op->call(input, other, alpha, inplace);
}

Maybe<one::Tensor> ScalarAdd(const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarAdd"));
  return __op->call(input, other, alpha);
}

Maybe<one::Tensor> Add(const TensorTuple& inputs, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&, bool>("Add"));
  return __op->call(inputs, inplace);
}

Maybe<one::Tensor> Amin(const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::vector<int32_t>>&, bool>("Amin"));
  return __op->call(input, dim, keepdim);
}

Maybe<one::Tensor> Sub(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("Sub"));
  return __op->call(input, other, alpha, inplace);
}

Maybe<one::Tensor> ScalarSub(const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const Scalar&, bool>("ScalarSub"));
  return __op->call(input, other, alpha, inplace);
}

Maybe<one::Tensor> ScalarSub(const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarSub"));
  return __op->call(input, other, alpha);
}

Maybe<one::Tensor> Mul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Mul"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarMul(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("ScalarMul"));
  return __op->call(input, other, inplace);
}

Maybe<one::Tensor> ScalarMul(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarMul"));
  return __op->call(input, other);
}

Maybe<one::Tensor> InplaceMul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("InplaceMul"));
  return __op->call(input, other);
}

Maybe<one::Tensor> InplaceScalarMul(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("InplaceScalarMul"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Addcmul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("Addcmul"));
  return __op->call(input, tensor1, tensor2, value);
}

Maybe<one::Tensor> InplaceAddcmul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("InplaceAddcmul"));
  return __op->call(input, tensor1, tensor2, value);
}

Maybe<one::Tensor> AddCDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("AddCDiv"));
  return __op->call(input, tensor1, tensor2, value);
}

Maybe<one::Tensor> InplaceAddCDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("InplaceAddCDiv"));
  return __op->call(input, tensor1, tensor2, value);
}

Maybe<one::Tensor> Div(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Div"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarDiv(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> InplaceDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("InplaceDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> InplaceScalarDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("InplaceScalarDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> DivGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& z, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("DivGrad"));
  return __op->call(dz, z, y);
}

Maybe<one::Tensor> BroadcastEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastNotEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastNotEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalNotEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalNotEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalNotEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalNotEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastGreater(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastGreater"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalGreater(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalGreater"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalGreater(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalGreater"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastGreaterEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastGreaterEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalGreaterEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalGreaterEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalGreaterEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalGreaterEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastLogicalAnd(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastLogicalAnd"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalAnd(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalAnd"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalAnd(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalAnd"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastLogicalOr(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastLogicalOr"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalOr(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalOr"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalOr(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalOr"));
  return __op->call(input, other);
}

Maybe<one::Tensor> LogicalNot(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("LogicalNot"));
  return __op->call(input);
}

Maybe<one::Tensor> BroadcastLogicalXor(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastLogicalXor"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalXor(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalXor"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalXor(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalXor"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastLess(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastLess"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalLess(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalLess"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalLess(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalLess"));
  return __op->call(input, other);
}

Maybe<one::Tensor> BroadcastLessEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastLessEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalLessEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarLogicalLessEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarLogicalLessEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarLogicalLessEqual"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Pow(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& exponent) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Pow"));
  return __op->call(input, exponent);
}

Maybe<one::Tensor> ScalarPow(const std::shared_ptr<one::Tensor>& input, const Scalar& exponent, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("ScalarPow"));
  return __op->call(input, exponent, inplace);
}

Maybe<one::Tensor> ScalarPow(const std::shared_ptr<one::Tensor>& input, const Scalar& exponent) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarPow"));
  return __op->call(input, exponent);
}

Maybe<one::Tensor> ScalarReversePow(const Scalar& exponent, const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const std::shared_ptr<one::Tensor>&>("ScalarReversePow"));
  return __op->call(exponent, input);
}

Maybe<one::Tensor> PowXGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("PowXGrad"));
  return __op->call(x, y, dz);
}

Maybe<one::Tensor> PowYGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("PowYGrad"));
  return __op->call(x, y, dz);
}

Maybe<one::Tensor> SearchSorted(const std::shared_ptr<one::Tensor>& sorted_sequence, const std::shared_ptr<one::Tensor>& values, bool out_int32, bool right) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, bool>("SearchSorted"));
  return __op->call(sorted_sequence, values, out_int32, right);
}

Maybe<one::Tensor> SearchSortedScalar(const std::shared_ptr<one::Tensor>& sorted_sequence, const Scalar& values, bool out_int32, bool right) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool, bool>("SearchSortedScalar"));
  return __op->call(sorted_sequence, values, out_int32, right);
}

Maybe<one::Tensor> ScalarPowGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& dy, const Scalar& exponent) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarPowGrad"));
  return __op->call(input, dy, exponent);
}

Maybe<one::Tensor> ScalarReversePowGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& dy, const Scalar& exponent) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarReversePowGrad"));
  return __op->call(input, dy, exponent);
}

Maybe<one::Tensor> BroadcastPow(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastPow"));
  return __op->call(x, y);
}

Maybe<one::Tensor> BroadcastPowXGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastPowXGrad"));
  return __op->call(x, y, dz);
}

Maybe<one::Tensor> BroadcastPowYGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastPowYGrad"));
  return __op->call(x, y, dz);
}

Maybe<one::Tensor> FloorDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FloorDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarFloorDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("ScalarFloorDiv"));
  return __op->call(input, other, inplace);
}

Maybe<one::Tensor> ScalarFloorDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarFloorDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> FloorDivXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FloorDivXGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> FloorDivYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FloorDivYGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> TruncDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TruncDiv"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarTruncDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("ScalarTruncDiv"));
  return __op->call(input, other, inplace);
}

Maybe<one::Tensor> TruncDivXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TruncDivXGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> TruncDivYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TruncDivYGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> XdivyXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("XdivyXGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> XdivyYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("XdivyYGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> XlogyXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("XlogyXGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> XlogyYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("XlogyYGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> Max(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Max"));
  return __op->call(input);
}

Maybe<one::TensorTuple> Max(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t, bool>("Max"));
  return __op->call(input, dim, keepdim);
}

Maybe<one::Tensor> Max(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Max"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Min(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Min"));
  return __op->call(input);
}

Maybe<one::TensorTuple> Min(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t, bool>("Min"));
  return __op->call(input, dim, keepdim);
}

Maybe<one::Tensor> Min(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Min"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Median(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Median"));
  return __op->call(input);
}

Maybe<one::TensorTuple> MedianWithIndices(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t, bool>("MedianWithIndices"));
  return __op->call(input, dim, keepdim);
}

Maybe<one::Tensor> ReduceMax(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMax"));
  return __op->call(x, axis, keepdim);
}

Maybe<one::Tensor> ReduceMin(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMin"));
  return __op->call(x, axis, keepdim);
}

Maybe<one::Tensor> ReduceSum(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceSum"));
  return __op->call(x, dim, keepdim);
}

Maybe<one::Tensor> ReduceSumWhole(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ReduceSumWhole"));
  return __op->call(x);
}

Maybe<one::Tensor> ReduceNanSum(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool, const Optional<Symbol<DType>>&>("ReduceNanSum"));
  return __op->call(input, dim, keepdim, dtype);
}

Maybe<one::Tensor> ReduceNanSumWhole(const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Symbol<DType>>&>("ReduceNanSumWhole"));
  return __op->call(input, dtype);
}

Maybe<one::Tensor> ReduceMean(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMean"));
  return __op->call(x, dim, keepdim);
}

Maybe<one::Tensor> ReduceMeanWhole(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ReduceMeanWhole"));
  return __op->call(x);
}

Maybe<one::Tensor> ReduceAll(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceAll"));
  return __op->call(x, dim, keepdim);
}

Maybe<one::Tensor> ReduceAllWhole(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ReduceAllWhole"));
  return __op->call(x);
}

Maybe<one::Tensor> ReduceAny(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceAny"));
  return __op->call(x, dim, keepdim);
}

Maybe<one::Tensor> ReduceAnyWhole(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ReduceAnyWhole"));
  return __op->call(x);
}

Maybe<one::Tensor> ReduceProd(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool, const Optional<Symbol<DType>>&>("ReduceProd"));
  return __op->call(x, dim, keepdim, dtype);
}

Maybe<one::Tensor> ReduceProdWhole(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Symbol<DType>>&>("ReduceProdWhole"));
  return __op->call(x, dtype);
}

Maybe<one::TensorTuple> ReduceMinDeviceStage(const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("ReduceMinDeviceStage"));
  return __op->call(in, axis);
}

Maybe<one::Tensor> ReduceMinDeviceStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& count, const std::vector<int32_t>& axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("ReduceMinDeviceStageGrad"));
  return __op->call(out_diff, mask, count, axis);
}

Maybe<one::TensorTuple> ReduceMaxDeviceStage(const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("ReduceMaxDeviceStage"));
  return __op->call(in, axis);
}

Maybe<one::Tensor> ReduceMaxDeviceStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& count, const std::vector<int32_t>& axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("ReduceMaxDeviceStageGrad"));
  return __op->call(out_diff, mask, count, axis);
}

Maybe<one::TensorTuple> ReduceMinGlobalStage(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMinGlobalStage"));
  return __op->call(in, device_count, axis, keepdims);
}

Maybe<one::Tensor> ReduceMinGlobalStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMinGlobalStageGrad"));
  return __op->call(out_diff, mask, device_count, axis, keepdims);
}

Maybe<one::TensorTuple> ReduceMaxGlobalStage(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMaxGlobalStage"));
  return __op->call(in, device_count, axis, keepdims);
}

Maybe<one::Tensor> ReduceMaxGlobalStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, bool>("ReduceMaxGlobalStageGrad"));
  return __op->call(out_diff, mask, device_count, axis, keepdims);
}

Maybe<one::Tensor> Transpose(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& perm) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("Transpose"));
  return __op->call(input, perm);
}

Maybe<one::Tensor> Transpose2dim(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("Transpose2dim"));
  return __op->call(input, dim0, dim1);
}

Maybe<one::Tensor> AsStrided(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size, const std::vector<int32_t>& stride, int32_t storage_offset) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t>("AsStrided"));
  return __op->call(input, size, stride, storage_offset);
}

Maybe<one::Tensor> AsStridedGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size, const std::vector<int32_t>& stride, int32_t storage_offset) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t>("AsStridedGrad"));
  return __op->call(dy, input, size, stride, storage_offset);
}

Maybe<one::Tensor> Select(const std::shared_ptr<one::Tensor>& input, int32_t dim, int32_t index) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("Select"));
  return __op->call(input, dim, index);
}

Maybe<one::Tensor> Swapaxes(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("Swapaxes"));
  return __op->call(input, dim0, dim1);
}

Maybe<one::Tensor> Swapdims(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("Swapdims"));
  return __op->call(input, dim0, dim1);
}

Maybe<one::Tensor> Amax(const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::vector<int32_t>>&, bool>("Amax"));
  return __op->call(input, dim, keepdim);
}

Maybe<one::Tensor> Permute(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("Permute"));
  return __op->call(input, dims);
}

Maybe<one::Tensor> TransposeAllDimProperty(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("TransposeAllDimProperty"));
  return __op->call(input);
}

Maybe<one::Tensor> TransposeAllDimFunction(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("TransposeAllDimFunction"));
  return __op->call(input);
}

Maybe<one::Tensor> NotEqualZero(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("NotEqualZero"));
  return __op->call(x);
}

Maybe<one::Tensor> NotEqualZeroGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("NotEqualZeroGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Reciprocal(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Reciprocal"));
  return __op->call(x);
}

Maybe<one::Tensor> ReciprocalGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReciprocalGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> ReciprocalNoNan(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ReciprocalNoNan"));
  return __op->call(x);
}

Maybe<one::Tensor> ReciprocalNoNanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReciprocalNoNanGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> ImageFlip(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& flip_code) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ImageFlip"));
  return __op->call(x, flip_code);
}

Maybe<one::Tensor> Sin(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sin"));
  return __op->call(x);
}

Maybe<one::Tensor> SinGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SinGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> SinGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SinGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Sin_(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sin_"));
  return __op->call(x);
}

Maybe<one::Tensor> Cos(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Cos"));
  return __op->call(x);
}

Maybe<one::Tensor> CosGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CosGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> CosGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CosGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Cosh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Cosh"));
  return __op->call(x);
}

Maybe<one::Tensor> CoshGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CoshGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> BroadcastFMod(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastFMod"));
  return __op->call(input, other);
}

Maybe<one::Tensor> ScalarFMod(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("ScalarFMod"));
  return __op->call(input, other, inplace);
}

Maybe<one::Tensor> ScalarFMod(const std::shared_ptr<one::Tensor>& input, const Scalar& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ScalarFMod"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Log(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Log"));
  return __op->call(x);
}

Maybe<one::Tensor> LogGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LogGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Log2(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Log2"));
  return __op->call(x);
}

Maybe<one::Tensor> Log2Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log2Grad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Log10(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Log10"));
  return __op->call(x);
}

Maybe<one::Tensor> Log10Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log10Grad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Sqrt(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sqrt"));
  return __op->call(x);
}

Maybe<one::Tensor> SqrtGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SqrtGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Rsqrt(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Rsqrt"));
  return __op->call(x);
}

Maybe<one::Tensor> RsqrtGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("RsqrtGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Square(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Square"));
  return __op->call(x);
}

Maybe<one::Tensor> SquareGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SquareGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> SqrtSquareSum(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("SqrtSquareSum"));
  return __op->call(x);
}

Maybe<one::Tensor> StandardDeviation(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::vector<int32_t>>&, const Optional<bool>&, const Optional<bool>&>("StandardDeviation"));
  return __op->call(x, dim, unbiased, keepdim);
}

Maybe<one::Tensor> Variance(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::vector<int32_t>>&, const Optional<bool>&, const Optional<bool>&>("Variance"));
  return __op->call(x, dim, unbiased, keepdim);
}

Maybe<one::Tensor> RMSLayerNormalization(const std::shared_ptr<one::Tensor>& hidden_states, const std::shared_ptr<one::Tensor>& weight, float variance_epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("RMSLayerNormalization"));
  return __op->call(hidden_states, weight, variance_epsilon);
}

Maybe<one::Tensor> Relu(const std::shared_ptr<one::Tensor>& x, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, bool>("Relu"));
  return __op->call(x, inplace);
}

Maybe<one::Tensor> ReluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReluGrad"));
  return __op->call(dy, y);
}

Maybe<one::Tensor> HannWindow(int64_t window_length, bool periodic, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, bool, const Optional<Symbol<Device>>&, const Optional<Symbol<DType>>&, bool>("HannWindow"));
  return __op->call(window_length, periodic, device, dtype, requires_grad);
}

Maybe<one::Tensor> GlobalHannWindow(int64_t window_length, bool periodic, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, bool, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, bool>("GlobalHannWindow"));
  return __op->call(window_length, periodic, placement, sbp, dtype, requires_grad);
}

Maybe<one::Tensor> HardTanh(const std::shared_ptr<one::Tensor>& x, double min_val, double max_val) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double>("HardTanh"));
  return __op->call(x, min_val, max_val);
}

Maybe<one::Tensor> HardTanhGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double min_val, double max_val) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double>("HardTanhGrad"));
  return __op->call(y, dy, min_val, max_val);
}

Maybe<one::Tensor> Tan(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Tan"));
  return __op->call(x);
}

Maybe<one::Tensor> TanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TanGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Tanh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Tanh"));
  return __op->call(x);
}

Maybe<one::Tensor> TanhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TanhGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Threshold(const std::shared_ptr<one::Tensor>& x, double threshold, double value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double>("Threshold"));
  return __op->call(x, threshold, value);
}

Maybe<one::Tensor> ThresholdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double threshold) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("ThresholdGrad"));
  return __op->call(x, dy, threshold);
}

Maybe<one::Tensor> Elu(const std::shared_ptr<one::Tensor>& x, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double>("Elu"));
  return __op->call(x, alpha);
}

Maybe<one::Tensor> EluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("EluGrad"));
  return __op->call(x, dy, alpha);
}

Maybe<one::Tensor> Celu(const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, bool>("Celu"));
  return __op->call(x, alpha, inplace);
}

Maybe<one::Tensor> CeluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("CeluGrad"));
  return __op->call(x, dy, alpha);
}

Maybe<one::Tensor> Gelu(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Gelu"));
  return __op->call(x);
}

Maybe<one::Tensor> GeluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("GeluGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> GeluWithApproximate(const std::shared_ptr<one::Tensor>& x, const std::string& approximate) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&>("GeluWithApproximate"));
  return __op->call(x, approximate);
}

Maybe<one::Tensor> Glu(const std::shared_ptr<one::Tensor>& input, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t>("Glu"));
  return __op->call(input, dim);
}

Maybe<one::Tensor> Sigmoid(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sigmoid"));
  return __op->call(x);
}

Maybe<one::Tensor> SigmoidGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SigmoidGrad"));
  return __op->call(y, dy);
}

Maybe<one::Tensor> HardSigmoid(const std::shared_ptr<one::Tensor>& input, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, bool>("HardSigmoid"));
  return __op->call(input, inplace);
}

Maybe<one::Tensor> HardSigmoidGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("HardSigmoidGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> HardShrink(const std::shared_ptr<one::Tensor>& x, double lambd, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, bool>("HardShrink"));
  return __op->call(x, lambd, inplace);
}

Maybe<one::Tensor> HardShrinkGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double lambd) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("HardShrinkGrad"));
  return __op->call(y, dy, lambd);
}

Maybe<one::Tensor> Softmax(const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<int64_t>&>("Softmax"));
  return __op->call(x, dim);
}

Maybe<one::Tensor> SoftmaxGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SoftmaxGrad"));
  return __op->call(dy, y);
}

Maybe<one::Tensor> LogSoftmax(const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<int64_t>&>("LogSoftmax"));
  return __op->call(x, dim);
}

Maybe<one::Tensor> LogSoftmaxGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LogSoftmaxGrad"));
  return __op->call(dy, y);
}

Maybe<one::Tensor> HardSwish(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("HardSwish"));
  return __op->call(x);
}

Maybe<one::Tensor> HardSwishGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("HardSwishGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> LeakyRelu(const std::shared_ptr<one::Tensor>& x, float alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, bool>("LeakyRelu"));
  return __op->call(x, alpha, inplace);
}

Maybe<one::Tensor> LeakyReluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, float alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("LeakyReluGrad"));
  return __op->call(x, dy, alpha);
}

Maybe<one::Tensor> Normal(float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, float, float, const Shape&, const Optional<one::Tensor>&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("Normal"));
  return __op->call(mean, std, size, out, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> Normal2(float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, float, float, int32_t, const Optional<one::Tensor>&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("Normal2"));
  return __op->call(mean, std, size, out, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalNormal(float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, float, float, const Shape&, const Optional<one::Tensor>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalNormal"));
  return __op->call(mean, std, size, out, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> GlobalNormal2(float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, float, float, int32_t, const Optional<one::Tensor>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalNormal2"));
  return __op->call(mean, std, size, out, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> Normalization(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, int32_t, float, float, bool>("Normalization"));
  return __op->call(x, moving_mean, moving_variance, gamma, beta, axis, epsilon, momentum, is_training);
}

Maybe<one::TensorTuple> NormalizationGrad(const std::shared_ptr<one::Tensor>& grad, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const std::shared_ptr<one::Tensor>& gamma, float epsilon, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, int32_t>("NormalizationGrad"));
  return __op->call(grad, x, mean, inv_variance, gamma, epsilon, axis);
}

Maybe<one::Tensor> NormalizationAddRelu(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& addend, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, float, float, bool>("NormalizationAddRelu"));
  return __op->call(x, addend, moving_mean, moving_variance, gamma, beta, axis, epsilon, momentum, is_training);
}

Maybe<one::TensorTuple> NormalizationAddReluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& moving_mean, const std::shared_ptr<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, const std::shared_ptr<one::Tensor>& reserve_space, const std::shared_ptr<one::Tensor>& y, int32_t axis, float epsilon, bool has_addend) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, float, bool>("NormalizationAddReluGrad"));
  return __op->call(x, dy, moving_mean, moving_variance, gamma, beta, reserve_space, y, axis, epsilon, has_addend);
}

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Scalar>&, const Symbol<DType>&, const Optional<Symbol<Device>>&, bool>("Eye"));
  return __op->call(n, m, dtype, device, requires_grad);
}

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const std::string& device, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Scalar>&, const Symbol<DType>&, const std::string&, bool>("Eye"));
  return __op->call(n, m, dtype, device, requires_grad);
}

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Scalar>&, const Symbol<DType>&, bool, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&>("Eye"));
  return __op->call(n, m, dtype, requires_grad, placement, sbp);
}

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const Symbol<SbpParallel>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Scalar>&, const Symbol<DType>&, bool, const Symbol<ParallelDesc>&, const Symbol<SbpParallel>&>("Eye"));
  return __op->call(n, m, dtype, requires_grad, placement, sbp);
}

Maybe<one::Tensor> EyeInplace(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("EyeInplace"));
  return __op->call(x);
}

Maybe<one::Tensor> Erfinv(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Erfinv"));
  return __op->call(x);
}

Maybe<one::Tensor> ErfinvInplace(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ErfinvInplace"));
  return __op->call(x);
}

Maybe<one::Tensor> Arange(const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Scalar&, const Scalar&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&>("Arange"));
  return __op->call(start, end, step, dtype, device);
}

Maybe<one::Tensor> Arange(const Scalar& end, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&>("Arange"));
  return __op->call(end, dtype, device);
}

Maybe<one::Tensor> GlobalArange(const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Scalar&, const Scalar&, const Optional<Symbol<DType>>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&>("GlobalArange"));
  return __op->call(start, end, step, dtype, placement, sbp);
}

Maybe<one::Tensor> GlobalArange(const Scalar& end, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Scalar&, const Optional<Symbol<DType>>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&>("GlobalArange"));
  return __op->call(end, dtype, placement, sbp);
}

Maybe<one::Tensor> Flatten(const std::shared_ptr<one::Tensor>& x, int32_t start_dim, int32_t end_dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("Flatten"));
  return __op->call(x, start_dim, end_dim);
}

Maybe<one::Tensor> ArgMax(const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<int32_t>&, const Optional<bool>&, const Optional<Symbol<DType>>&>("ArgMax"));
  return __op->call(x, dim, keepdim, dtype);
}

Maybe<one::Tensor> ArgMin(const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<int32_t>&, const Optional<bool>&, const Optional<Symbol<DType>>&>("ArgMin"));
  return __op->call(x, dim, keepdim, dtype);
}

Maybe<one::TensorTuple> ArgWhere(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const Symbol<DType>&>("ArgWhere"));
  return __op->call(x, dtype);
}

Maybe<one::TensorTuple> NonZero(const std::shared_ptr<one::Tensor>& x, bool as_tuple) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, bool>("NonZero"));
  return __op->call(x, as_tuple);
}

Maybe<one::Tensor> BroadcastLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& broadcast_axes) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("BroadcastLike"));
  return __op->call(x, like, broadcast_axes);
}

Maybe<one::Tensor> Cast(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype, bool pin_memory) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<DType>&, bool>("Cast"));
  return __op->call(x, dtype, pin_memory);
}

Maybe<one::Tensor> Constant(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Scalar&, const Symbol<DType>&, const Optional<Symbol<Device>>&>("Constant"));
  return __op->call(shape, value, dtype, device);
}

Maybe<one::Tensor> GlobalConstant(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Scalar&, const Symbol<DType>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&>("GlobalConstant"));
  return __op->call(shape, value, dtype, placement, sbp);
}

Maybe<one::Tensor> Empty(const Shape& shape, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool pin_memory) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Symbol<DType>&, const Optional<Symbol<Device>>&, bool>("Empty"));
  return __op->call(shape, dtype, device, pin_memory);
}

Maybe<one::Tensor> GlobalEmpty(const Shape& shape, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Symbol<DType>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&>("GlobalEmpty"));
  return __op->call(shape, dtype, placement, sbp);
}

Maybe<one::Tensor> ZerosLike(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ZerosLike"));
  return __op->call(x);
}

Maybe<one::Tensor> OnesLike(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("OnesLike"));
  return __op->call(x);
}

Maybe<one::Tensor> Bernoulli(const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<DType>&, const Optional<one::Generator>&, bool>("Bernoulli"));
  return __op->call(input, dtype, generator, inplace);
}

Maybe<one::Tensor> BernoulliProb(const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, const Symbol<DType>&, const Optional<one::Generator>&, bool>("BernoulliProb"));
  return __op->call(input, p, dtype, generator, inplace);
}

Maybe<one::Tensor> BernoulliInplace(const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<DType>&, const Optional<one::Generator>&>("BernoulliInplace"));
  return __op->call(input, dtype, generator);
}

Maybe<one::Tensor> BernoulliProbInplace(const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, const Symbol<DType>&, const Optional<one::Generator>&>("BernoulliProbInplace"));
  return __op->call(input, p, dtype, generator);
}

Maybe<one::Tensor> Concat(const TensorTuple& inputs, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&, int64_t>("Concat"));
  return __op->call(inputs, dim);
}

Maybe<one::Tensor> BiasAdd(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& bias, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("BiasAdd"));
  return __op->call(x, bias, axis);
}

Maybe<one::Tensor> Conv1d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::string&>("Conv1d"));
  return __op->call(x, weight, bias, stride, padding, dilation, groups, channel_pos);
}

Maybe<one::Tensor> Conv2d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::string&>("Conv2d"));
  return __op->call(x, weight, bias, stride, padding, dilation, groups, channel_pos);
}

Maybe<one::Tensor> Conv3d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::string&>("Conv3d"));
  return __op->call(x, weight, bias, stride, padding, dilation, groups, channel_pos);
}

Maybe<one::Tensor> FakeQuantization(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int32_t, const std::string&>("FakeQuantization"));
  return __op->call(in, scale, zero_point, quantization_formula, quantization_bit, quantization_scheme);
}

Maybe<one::Tensor> Quantization(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int32_t, const std::string&>("Quantization"));
  return __op->call(in, scale, zero_point, quantization_formula, quantization_bit, quantization_scheme);
}

Maybe<one::TensorTuple> MinMaxObserver(const std::shared_ptr<one::Tensor>& in, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, bool per_layer_quantization) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::string&, int32_t, const std::string&, bool>("MinMaxObserver"));
  return __op->call(in, quantization_formula, quantization_bit, quantization_scheme, per_layer_quantization);
}

Maybe<one::TensorTuple> MovingAverageMinMaxObserver(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& current_train_step, const std::shared_ptr<one::Tensor>& moving_max, const std::shared_ptr<one::Tensor>& moving_min, bool training, int64_t stop_update_after_iters, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, float momentum) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, int64_t, const std::string&, int32_t, const std::string&, float>("MovingAverageMinMaxObserver"));
  return __op->call(in, current_train_step, moving_max, moving_min, training, stop_update_after_iters, quantization_formula, quantization_bit, quantization_scheme, momentum);
}

Maybe<one::Tensor> ConvDataGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x, int32_t num_spatial_dims, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& strides, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& dilation_rate, int32_t groups, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::string&>("ConvDataGrad"));
  return __op->call(dy, weight, x, num_spatial_dims, kernel_size, strides, padding_before, dilation_rate, groups, data_format);
}

Maybe<one::Tensor> ConvFilterGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, int32_t num_spatial_dims, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& strides, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& dilation_rate, int32_t groups, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::string&>("ConvFilterGrad"));
  return __op->call(dy, x, num_spatial_dims, kernel_size, strides, padding_before, dilation_rate, groups, data_format);
}

Maybe<one::Tensor> ConvBiasGrad(const std::shared_ptr<one::Tensor>& dy, int32_t num_spatial_dims, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::string&>("ConvBiasGrad"));
  return __op->call(dy, num_spatial_dims, data_format);
}

Maybe<one::Tensor> Deconv1d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::vector<int32_t>&, const std::string&>("Deconv1d"));
  return __op->call(x, weight, bias, stride, padding, output_padding, groups, dilation, data_format);
}

Maybe<one::Tensor> Deconv2d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::vector<int32_t>&, const std::string&>("Deconv2d"));
  return __op->call(x, weight, bias, stride, padding, output_padding, groups, dilation, data_format);
}

Maybe<one::Tensor> Deconv3d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, int32_t, const std::vector<int32_t>&, const std::string&>("Deconv3d"));
  return __op->call(x, weight, bias, stride, padding, output_padding, groups, dilation, data_format);
}

Maybe<one::Tensor> Expand(const std::shared_ptr<one::Tensor>& x, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&>("Expand"));
  return __op->call(x, shape);
}

Maybe<one::Tensor> Repeat(const std::shared_ptr<one::Tensor>& input, const Shape& repeat_shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&>("Repeat"));
  return __op->call(input, repeat_shape);
}

Maybe<one::Tensor> RepeatInterLeaveIndex(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& cumsum, int32_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("RepeatInterLeaveIndex"));
  return __op->call(input, cumsum, dim);
}

Maybe<one::Tensor> RepeatInterLeaveInt(const std::shared_ptr<one::Tensor>& input, int32_t repeats, const Optional<int32_t>& dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const Optional<int32_t>&>("RepeatInterLeaveInt"));
  return __op->call(input, repeats, dim);
}

Maybe<one::Tensor> RepeatInterLeaveTensor(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& repeats, int32_t dim, const Optional<int32_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, const Optional<int32_t>&>("RepeatInterLeaveTensor"));
  return __op->call(input, repeats, dim, output_size);
}

Maybe<one::Tensor> Tile(const std::shared_ptr<one::Tensor>& input, const Shape& dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&>("Tile"));
  return __op->call(input, dims);
}

Maybe<one::Tensor> Roll(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& shifts, const Optional<std::vector<int32_t>>& dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&>("Roll"));
  return __op->call(x, shifts, dims);
}

Maybe<one::Tensor> ExpandDims(const std::shared_ptr<one::Tensor>& input, int32_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("ExpandDims"));
  return __op->call(input, dim);
}

Maybe<one::Tensor> Unsqueeze(const std::shared_ptr<one::Tensor>& input, int32_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("Unsqueeze"));
  return __op->call(input, dim);
}

Maybe<one::Tensor> UnsqueezeMultiple(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dim, int32_t dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, int32_t>("UnsqueezeMultiple"));
  return __op->call(input, dim, dims);
}

Maybe<one::Tensor> Squeeze(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::vector<int32_t>>&>("Squeeze"));
  return __op->call(x, dim);
}

Maybe<one::Tensor> Exp(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Exp"));
  return __op->call(x);
}

Maybe<one::Tensor> ExpGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ExpGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Gather(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& indices, int64_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("Gather"));
  return __op->call(x, indices, axis);
}

Maybe<one::Tensor> DimGather(const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index, bool sparse_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const std::shared_ptr<one::Tensor>&, bool>("DimGather"));
  return __op->call(input, dim, index, sparse_grad);
}

Maybe<one::Tensor> EmbeddingReNorm(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices, double max_norm, double norm_type) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double>("EmbeddingReNorm"));
  return __op->call(in, indices, max_norm, norm_type);
}

Maybe<one::Tensor> Embedding(const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& indices, const Optional<int64_t>& padding_idx, bool scale_grad_by_freq) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<int64_t>&, bool>("Embedding"));
  return __op->call(weight, indices, padding_idx, scale_grad_by_freq);
}

Maybe<one::Tensor> EmbeddingGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& indices, int64_t padding_idx, bool scale_grad_by_freq) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, bool>("EmbeddingGrad"));
  return __op->call(dy, weight, indices, padding_idx, scale_grad_by_freq);
}

Maybe<one::Tensor> ArgSort(const std::shared_ptr<one::Tensor>& in, const std::string& direction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&>("ArgSort"));
  return __op->call(in, direction);
}

Maybe<one::Tensor> GatherNd(const std::shared_ptr<one::Tensor>& params, const std::shared_ptr<one::Tensor>& indices) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("GatherNd"));
  return __op->call(params, indices);
}

Maybe<one::Tensor> ScatterNd(const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Shape&>("ScatterNd"));
  return __op->call(indices, updates, shape);
}

Maybe<one::Tensor> TensorScatterNdUpdate(const std::shared_ptr<one::Tensor>& tensor, const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("TensorScatterNdUpdate"));
  return __op->call(tensor, indices, updates, inplace);
}

Maybe<one::Tensor> ScatterNdLike(const std::shared_ptr<one::Tensor>& like, const std::shared_ptr<one::Tensor>& updates, const std::shared_ptr<one::Tensor>& indices) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ScatterNdLike"));
  return __op->call(like, updates, indices);
}

Maybe<one::Tensor> MatMul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, bool transpose_a, bool transpose_b, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, bool, double>("MatMul"));
  return __op->call(input, other, transpose_a, transpose_b, alpha);
}

Maybe<one::Tensor> MatMulNoBroadCast(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mat2) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MatMulNoBroadCast"));
  return __op->call(input, mat2);
}

Maybe<one::Tensor> FusedMLP(const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const TensorTuple&, bool>("FusedMLP"));
  return __op->call(x, weights, biases, skip_final_activation);
}

Maybe<one::TensorTuple> FusedMLPGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& cublas_aux, const TensorTuple& hidden, const std::vector<float>& alpha_list) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const TensorTuple&, const TensorTuple&, const std::vector<float>&>("FusedMLPGrad"));
  return __op->call(dy, x, weights, cublas_aux, hidden, alpha_list);
}

Maybe<one::TensorTuple> CublasBiasAddReluMatmulGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& aux, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("CublasBiasAddReluMatmulGrad"));
  return __op->call(dy, weight, aux, alpha);
}

Maybe<one::TensorTuple> CublasMatmulBiasAddGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CublasMatmulBiasAddGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> FusedMatmulBiasAddReluDropout(const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation, const std::vector<float>& dropout_rate_list, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const TensorTuple&, bool, const std::vector<float>&, const Optional<one::Generator>&>("FusedMatmulBiasAddReluDropout"));
  return __op->call(x, weights, biases, skip_final_activation, dropout_rate_list, generator);
}

Maybe<one::Tensor> FusedReluDropoutGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("FusedReluDropoutGrad"));
  return __op->call(dy, mask, scale);
}

Maybe<one::Tensor> BroadcastMatmulGradB(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("BroadcastMatmulGradB"));
  return __op->call(a, b, alpha);
}

Maybe<one::Tensor> BatchMatMul(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, bool transpose_a, bool transpose_b, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, bool, double>("BatchMatMul"));
  return __op->call(a, b, transpose_a, transpose_b, alpha);
}

Maybe<one::Tensor> MatrixVectorProduct(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& vec) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MatrixVectorProduct"));
  return __op->call(input, vec);
}

Maybe<one::Tensor> MatrixVectorProductGradA(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& b) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MatrixVectorProductGradA"));
  return __op->call(dy, b);
}

Maybe<one::Tensor> MatrixVectorProductGradB(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& a) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MatrixVectorProductGradB"));
  return __op->call(dy, a);
}

Maybe<one::Tensor> VectorMatrixProduct(const std::shared_ptr<one::Tensor>& vec, const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("VectorMatrixProduct"));
  return __op->call(vec, input);
}

Maybe<one::Tensor> VectorMatrixProductGradA(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& b) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("VectorMatrixProductGradA"));
  return __op->call(dy, b);
}

Maybe<one::Tensor> VectorMatrixProductGradB(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& a) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("VectorMatrixProductGradB"));
  return __op->call(dy, a);
}

Maybe<one::Tensor> TensorDot(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, const std::vector<int32_t>& dims_a, const std::vector<int32_t>& dims_b) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&>("TensorDot"));
  return __op->call(a, b, dims_a, dims_b);
}

Maybe<one::Tensor> TensorDotIntDims(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("TensorDotIntDims"));
  return __op->call(a, b, dims);
}

Maybe<one::Tensor> L1Loss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&>("L1Loss"));
  return __op->call(input, target, reduction);
}

Maybe<one::Tensor> MseLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&>("MseLoss"));
  return __op->call(input, target, reduction);
}

Maybe<one::Tensor> KLDivLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, const std::string&>("KLDivLoss"));
  return __op->call(input, target, log_target, reduction);
}

Maybe<one::Tensor> KLDivLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("KLDivLossGrad"));
  return __op->call(dy, input, target, log_target);
}

Maybe<one::Tensor> KLDivLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("KLDivLossTargetGrad"));
  return __op->call(dy, input, target, log_target);
}

Maybe<one::Tensor> NLLLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int64_t, const std::string&>("NLLLoss"));
  return __op->call(input, target, weight, ignore_index, reduction);
}

Maybe<one::Tensor> NLLGrad(const std::shared_ptr<one::Tensor>& out_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int64_t>("NLLGrad"));
  return __op->call(out_grad, input, target, weight, ignore_index);
}

Maybe<one::Tensor> BinaryCrossEntropyLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::string&>("BinaryCrossEntropyLoss"));
  return __op->call(input, target, weight, reduction);
}

Maybe<one::Tensor> BinaryCrossEntropyLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&>("BinaryCrossEntropyLossGrad"));
  return __op->call(dy, input, target, weight);
}

Maybe<one::Tensor> BinaryCrossEntropyLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&>("BinaryCrossEntropyLossTargetGrad"));
  return __op->call(dy, input, target, weight);
}

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const std::string&>("BinaryCrossEntropyWithLogitsLoss"));
  return __op->call(input, target, weight, pos_weight, reduction);
}

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("BinaryCrossEntropyWithLogitsLossGrad"));
  return __op->call(dy, input, target, weight, pos_weight);
}

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("BinaryCrossEntropyWithLogitsLossTargetGrad"));
  return __op->call(dy, input, target, weight, pos_weight);
}

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsReduceMeanLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BinaryCrossEntropyWithLogitsReduceMeanLossGrad"));
  return __op->call(dy, input, target);
}

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsReduceMeanLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BinaryCrossEntropyWithLogitsReduceMeanLossTargetGrad"));
  return __op->call(dy, input, target);
}

Maybe<one::Tensor> SparseCrossEntropy(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseCrossEntropy"));
  return __op->call(prediction, label, depth);
}

Maybe<one::Tensor> SparseCrossEntropyGrad(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& dy, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseCrossEntropyGrad"));
  return __op->call(prediction, label, dy, depth);
}

Maybe<one::Tensor> SparseCrossEntropyMs(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseCrossEntropyMs"));
  return __op->call(prediction, label, depth);
}

Maybe<one::Tensor> CrossEntropy(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction, double label_smoothing) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int64_t, const std::string&, double>("CrossEntropy"));
  return __op->call(input, target, weight, ignore_index, reduction, label_smoothing);
}

Maybe<one::Tensor> CrossEntropyLabelSmoothing(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction, double label_smoothing) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int64_t, const std::string&, double>("CrossEntropyLabelSmoothing"));
  return __op->call(input, target, weight, ignore_index, reduction, label_smoothing);
}

Maybe<one::Tensor> CrossEntropyProb(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const std::string& reduction, double label_smoothing) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const std::string&, double>("CrossEntropyProb"));
  return __op->call(input, target, weight, reduction, label_smoothing);
}

Maybe<one::Tensor> SparseCrossEntropyMsGrad(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& dy, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseCrossEntropyMsGrad"));
  return __op->call(prediction, label, dy, depth);
}

Maybe<one::Tensor> SparseSoftmaxCrossEntropy(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SparseSoftmaxCrossEntropy"));
  return __op->call(logits, label);
}

Maybe<one::Tensor> SparseSoftmaxCrossEntropyGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& prob, const std::shared_ptr<one::Tensor>& label, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseSoftmaxCrossEntropyGrad"));
  return __op->call(dy, prob, label, depth);
}

Maybe<one::Tensor> SparseSoftmaxCrossEntropyMsGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& prob, const std::shared_ptr<one::Tensor>& label, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("SparseSoftmaxCrossEntropyMsGrad"));
  return __op->call(dy, prob, label, depth);
}

Maybe<one::Tensor> SoftmaxCrossEntropy(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SoftmaxCrossEntropy"));
  return __op->call(logits, label);
}

Maybe<one::Tensor> SoftmaxCrossEntropyGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& prob) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SoftmaxCrossEntropyGrad"));
  return __op->call(dy, label, prob);
}

Maybe<one::Tensor> SmoothL1Loss(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label, float beta, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, const std::string&>("SmoothL1Loss"));
  return __op->call(logits, label, beta, reduction);
}

Maybe<one::Tensor> SmoothL1LossGrad(const std::shared_ptr<one::Tensor>& loss_grad, const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, float beta) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("SmoothL1LossGrad"));
  return __op->call(loss_grad, prediction, label, beta);
}

Maybe<one::Tensor> CombinedMarginLoss(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& label, float m1, float m2, float m3) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float, float>("CombinedMarginLoss"));
  return __op->call(x, label, m1, m2, m3);
}

Maybe<one::Tensor> CombinedMarginLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& theta, float m1, float m2, float m3, int64_t depth) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float, float, int64_t>("CombinedMarginLossGrad"));
  return __op->call(dy, label, theta, m1, m2, m3, depth);
}

Maybe<one::Tensor> TripletMarginLoss(const std::shared_ptr<one::Tensor>& anchor, const std::shared_ptr<one::Tensor>& positive, const std::shared_ptr<one::Tensor>& negative, float margin, float p, float eps, bool swap, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float, float, bool, const std::string&>("TripletMarginLoss"));
  return __op->call(anchor, positive, negative, margin, p, eps, swap, reduction);
}

Maybe<one::Tensor> MarginRankingLoss(const std::shared_ptr<one::Tensor>& input_1, const std::shared_ptr<one::Tensor>& input_2, const std::shared_ptr<one::Tensor>& target, float margin, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, const std::string&>("MarginRankingLoss"));
  return __op->call(input_1, input_2, target, margin, reduction);
}

Maybe<one::Tensor> CtcLoss(const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& input_lengths, const std::shared_ptr<one::Tensor>& target_lengths, int64_t max_target_length, int64_t blank, bool zero_infinity, const std::string& reduction) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, bool, const std::string&>("CtcLoss"));
  return __op->call(log_probs, targets, input_lengths, target_lengths, max_target_length, blank, zero_infinity, reduction);
}

Maybe<one::Tensor> AffineGrid(const std::shared_ptr<one::Tensor>& theta, const Shape& size, bool align_corners) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&, bool>("AffineGrid"));
  return __op->call(theta, size, align_corners);
}

Maybe<one::Tensor> AffineGridGrad(const std::shared_ptr<one::Tensor>& dgrid, const Shape& size, bool align_corners) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&, bool>("AffineGridGrad"));
  return __op->call(dgrid, size, align_corners);
}

Maybe<one::Tensor> GridSample(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& grid, const std::string& interpolation_mode, const std::string& padding_mode, bool align_corners) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, const std::string&, bool>("GridSample"));
  return __op->call(input, grid, interpolation_mode, padding_mode, align_corners);
}

Maybe<one::TensorTuple> GridSampleGrad(const std::shared_ptr<one::Tensor>& doutput, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& grid, const std::string& interpolation_mode, const std::string& padding_mode, bool align_corners) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, const std::string&, bool>("GridSampleGrad"));
  return __op->call(doutput, input, grid, interpolation_mode, padding_mode, align_corners);
}

Maybe<one::Tensor> Where(const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Where"));
  return __op->call(condition, x, y);
}

Maybe<one::Tensor> WhereScalarX(const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const std::shared_ptr<one::Tensor>&>("WhereScalarX"));
  return __op->call(condition, x, y);
}

Maybe<one::Tensor> WhereScalarY(const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const Scalar& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("WhereScalarY"));
  return __op->call(condition, x, y);
}

Maybe<one::Tensor> WhereScalarXY(const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const Scalar& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const Scalar&>("WhereScalarXY"));
  return __op->call(condition, x, y);
}

Maybe<one::Tensor> MaskedFill(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("MaskedFill"));
  return __op->call(input, mask, value);
}

Maybe<one::Tensor> MaskedFillInplace(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Scalar&>("MaskedFillInplace"));
  return __op->call(input, mask, value);
}

Maybe<one::Tensor> MovedimInt(const std::shared_ptr<one::Tensor>& input, int32_t source, int32_t destination) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("MovedimInt"));
  return __op->call(input, source, destination);
}

Maybe<one::Tensor> MovedimVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& source, const std::vector<int32_t>& destination) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&>("MovedimVec"));
  return __op->call(input, source, destination);
}

Maybe<one::TensorTuple> TensorSplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections, int32_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t>("TensorSplitInt"));
  return __op->call(input, indices_or_sections, dim);
}

Maybe<one::TensorTuple> TensorSplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections, int32_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, int32_t>("TensorSplitVec"));
  return __op->call(input, indices_or_sections, dim);
}

Maybe<one::TensorTuple> HsplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t>("HsplitInt"));
  return __op->call(input, indices_or_sections);
}

Maybe<one::TensorTuple> HsplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("HsplitVec"));
  return __op->call(input, indices_or_sections);
}

Maybe<one::TensorTuple> VsplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t>("VsplitInt"));
  return __op->call(input, indices_or_sections);
}

Maybe<one::TensorTuple> VsplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("VsplitVec"));
  return __op->call(input, indices_or_sections);
}

Maybe<one::Tensor> Negative(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Negative"));
  return __op->call(x);
}

Maybe<one::Tensor> LayerNormAffine(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, double>("LayerNormAffine"));
  return __op->call(x, gamma, beta, begin_norm_axis, begin_params_axis, epsilon);
}

Maybe<one::Tensor> LayerNorm(const std::shared_ptr<one::Tensor>& x, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, double>("LayerNorm"));
  return __op->call(x, begin_norm_axis, begin_params_axis, epsilon);
}

Maybe<one::Tensor> LayerNormGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, int64_t begin_norm_axis, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, double>("LayerNormGrad"));
  return __op->call(dy, x, mean, inv_variance, begin_norm_axis, epsilon);
}

Maybe<one::Tensor> LayerNormAffineGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const std::shared_ptr<one::Tensor>& gamma, int64_t begin_norm_axis, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, double>("LayerNormAffineGrad"));
  return __op->call(dy, x, mean, inv_variance, gamma, begin_norm_axis, epsilon);
}

Maybe<one::TensorTuple> LayerNormParamGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, int64_t begin_params_axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("LayerNormParamGrad"));
  return __op->call(dy, x, mean, inv_variance, begin_params_axis);
}

Maybe<one::Tensor> GroupNorm(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, bool affine, int32_t num_groups, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, bool, int32_t, double>("GroupNorm"));
  return __op->call(x, gamma, beta, affine, num_groups, epsilon);
}

Maybe<one::Tensor> GroupNormGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const Optional<one::Tensor>& gamma, int32_t num_groups, double epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int32_t, double>("GroupNormGrad"));
  return __op->call(dy, x, mean, inv_variance, gamma, num_groups, epsilon);
}

Maybe<one::TensorTuple> GroupNormParamGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("GroupNormParamGrad"));
  return __op->call(dy, x, mean, inv_variance);
}

Maybe<one::Tensor> TFAvgPool2D(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, const std::string& padding, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& padding_after, const std::string& data_format, bool ceil_mode) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::string&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::string&, bool>("TFAvgPool2D"));
  return __op->call(x, kernel_size, stride, padding, padding_before, padding_after, data_format, ceil_mode);
}

Maybe<one::Tensor> CtcLossGrad(const std::shared_ptr<one::Tensor>& loss_grad, const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& input_lengths, const std::shared_ptr<one::Tensor>& target_lengths, const std::shared_ptr<one::Tensor>& loss, const std::shared_ptr<one::Tensor>& alpha, int64_t blank, bool zero_infinity, int64_t max_target_length) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, bool, int64_t>("CtcLossGrad"));
  return __op->call(loss_grad, log_probs, targets, input_lengths, target_lengths, loss, alpha, blank, zero_infinity, max_target_length);
}

Maybe<one::Tensor> AdaptiveAvgPool1D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveAvgPool1D"));
  return __op->call(x, output_size);
}

Maybe<one::Tensor> AdaptiveAvgPool2D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveAvgPool2D"));
  return __op->call(x, output_size);
}

Maybe<one::Tensor> AdaptiveAvgPool3D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveAvgPool3D"));
  return __op->call(x, output_size);
}

Maybe<one::Tensor> AdaptivePoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, const std::string& mode, int32_t ndims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int32_t>("AdaptivePoolNdGrad"));
  return __op->call(x, dy, mode, ndims);
}

Maybe<one::Tensor> TFPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, const std::string& mode, int32_t ndims, const std::string& data_format, const std::string& padding, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& padding_after, const std::vector<int32_t>& pool_size, const std::vector<int32_t>& strides, bool ceil_mode) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int32_t, const std::string&, const std::string&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool>("TFPoolNdGrad"));
  return __op->call(x, y, dy, mode, ndims, data_format, padding, padding_before, padding_after, pool_size, strides, ceil_mode);
}

Maybe<one::TensorTuple> MaxPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool, bool, const std::string&>("MaxPool1D"));
  return __op->call(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode, data_format);
}

Maybe<one::TensorTuple> MaxPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool, bool, const std::string&>("MaxPool2D"));
  return __op->call(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode, data_format);
}

Maybe<one::TensorTuple> MaxPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool, bool, const std::string&>("MaxPool3D"));
  return __op->call(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode, data_format);
}

Maybe<one::Tensor> MaxPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& indice, const std::shared_ptr<one::Tensor>& dy, int32_t ndims, const std::string& data_format, const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, const std::string&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool, bool>("MaxPoolNdGrad"));
  return __op->call(x, indice, dy, ndims, data_format, padding, kernel_size, stride, dilation, return_indices, ceil_mode);
}

Maybe<one::Tensor> PRelu(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("PRelu"));
  return __op->call(x, alpha);
}

Maybe<one::TensorTuple> PReluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("PReluGrad"));
  return __op->call(dy, x, alpha);
}

Maybe<one::Tensor> Reshape(const std::shared_ptr<one::Tensor>& x, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&>("Reshape"));
  return __op->call(x, shape);
}

Maybe<one::Tensor> View(const std::shared_ptr<one::Tensor>& x, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&>("View"));
  return __op->call(x, shape);
}

Maybe<one::Tensor> ToContiguous(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("ToContiguous"));
  return __op->call(input);
}

Maybe<one::Tensor> InplaceToContiguous(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("InplaceToContiguous"));
  return __op->call(input);
}

Maybe<one::Tensor> SliceView1dContiguous(const std::shared_ptr<one::Tensor>& x, int64_t start, int64_t end) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t>("SliceView1dContiguous"));
  return __op->call(x, start, end);
}

Maybe<one::Tensor> Narrow(const std::shared_ptr<one::Tensor>& input, int64_t dim, int64_t start, int64_t length) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, int64_t>("Narrow"));
  return __op->call(input, dim, start, length);
}

Maybe<one::Tensor> NarrowGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& like, int64_t dim, int64_t start, int64_t length) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, int64_t>("NarrowGrad"));
  return __op->call(dy, like, dim, start, length);
}

Maybe<one::Tensor> Slice(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, const Optional<bool>& enable_view_slice) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const Optional<bool>&>("Slice"));
  return __op->call(x, start, stop, step, enable_view_slice);
}

Maybe<one::Tensor> SliceUpdate(const std::shared_ptr<one::Tensor>& ref, const std::shared_ptr<one::Tensor>& value, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, bool>("SliceUpdate"));
  return __op->call(ref, value, start, stop, step, inplace);
}

Maybe<one::Tensor> SliceGrad(const std::shared_ptr<one::Tensor>& dy, const Shape& like_shape, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&>("SliceGrad"));
  return __op->call(dy, like_shape, start, stop, step);
}

Maybe<one::Tensor> Copy(const std::shared_ptr<one::Tensor>& x, const std::string& device_type, int64_t device_id, bool pin_memory) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&, int64_t, bool>("Copy"));
  return __op->call(x, device_type, device_id, pin_memory);
}

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device, const Optional<Symbol<DType>>& dtype, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::string>&, const Optional<Symbol<DType>>&, bool>("To"));
  return __op->call(x, device, dtype, copy);
}

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Symbol<Device>>&, const Optional<Symbol<DType>>&, bool>("To"));
  return __op->call(x, device, dtype, copy);
}

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Symbol<DType>>&, bool>("To"));
  return __op->call(x, dtype, copy);
}

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& other, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("To"));
  return __op->call(x, other, copy);
}

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<std::string>&>("To"));
  return __op->call(x, device);
}

Maybe<one::Tensor> Flip(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("Flip"));
  return __op->call(x, dims);
}

Maybe<one::Tensor> Upsample(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const std::string& interpolation, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, bool, const std::string&, const std::string&>("Upsample"));
  return __op->call(x, height_scale, width_scale, align_corners, interpolation, data_format);
}

Maybe<one::Tensor> UpsampleGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const std::string& data_format, const std::string& interpolation) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, bool, const std::string&, const std::string&>("UpsampleGrad"));
  return __op->call(dy, x, height_scale, width_scale, align_corners, data_format, interpolation);
}

Maybe<one::Tensor> UpsampleLinear1D(const std::shared_ptr<one::Tensor>& x, double scale_factor, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleLinear1D"));
  return __op->call(x, scale_factor, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleLinear1DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double scale_factor, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleLinear1DGrad"));
  return __op->call(dy, x, scale_factor, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest1D(const std::shared_ptr<one::Tensor>& x, double scale_factor, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest1D"));
  return __op->call(x, scale_factor, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest1DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double scale_factor, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest1DGrad"));
  return __op->call(dy, x, scale_factor, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest2D"));
  return __op->call(x, height_scale, width_scale, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest2DGrad"));
  return __op->call(dy, x, height_scale, width_scale, output_size, data_format);
}

Maybe<one::Tensor> UpsampleBilinear2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleBilinear2D"));
  return __op->call(x, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleBilinear2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleBilinear2DGrad"));
  return __op->call(dy, x, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleBicubic2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleBicubic2D"));
  return __op->call(x, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleBicubic2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleBicubic2DGrad"));
  return __op->call(dy, x, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest3D(const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest3D"));
  return __op->call(x, depth_scale, height_scale, width_scale, output_size, data_format);
}

Maybe<one::Tensor> UpsampleNearest3DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, double, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleNearest3DGrad"));
  return __op->call(dy, x, depth_scale, height_scale, width_scale, output_size, data_format);
}

Maybe<one::Tensor> UpsampleTrilinear3D(const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleTrilinear3D"));
  return __op->call(x, depth_scale, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> UpsampleTrilinear3DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double, double, bool, const Optional<std::vector<int64_t>>&, const std::string&>("UpsampleTrilinear3DGrad"));
  return __op->call(dy, x, depth_scale, height_scale, width_scale, align_corners, output_size, data_format);
}

Maybe<one::Tensor> Abs(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Abs"));
  return __op->call(x);
}

Maybe<one::Tensor> AbsGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AbsGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Acos(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Acos"));
  return __op->call(x);
}

Maybe<one::Tensor> AcosGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AcosGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Acosh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Acosh"));
  return __op->call(x);
}

Maybe<one::Tensor> AcoshGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AcoshGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Asin(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Asin"));
  return __op->call(x);
}

Maybe<one::Tensor> AsinGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AsinGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Asinh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Asinh"));
  return __op->call(x);
}

Maybe<one::Tensor> AsinhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AsinhGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Atan(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Atan"));
  return __op->call(x);
}

Maybe<one::Tensor> AtanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AtanGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Atan2(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Atan2"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Atan2XGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Atan2XGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> Atan2YGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Atan2YGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> Atanh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Atanh"));
  return __op->call(x);
}

Maybe<one::Tensor> AtanhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AtanhGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Ceil(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Ceil"));
  return __op->call(x);
}

Maybe<one::Tensor> CeilGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CeilGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Erf(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Erf"));
  return __op->call(x);
}

Maybe<one::Tensor> ErfGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ErfGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Erfc(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Erfc"));
  return __op->call(x);
}

Maybe<one::Tensor> ErfcGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ErfcGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Expm1(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Expm1"));
  return __op->call(x);
}

Maybe<one::Tensor> Expm1Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Expm1Grad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Floor(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Floor"));
  return __op->call(x);
}

Maybe<one::Tensor> Floor_(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Floor_"));
  return __op->call(x);
}

Maybe<one::Tensor> FloorGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FloorGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Lgamma(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Lgamma"));
  return __op->call(x);
}

Maybe<one::Tensor> LgammaGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LgammaGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Log1p(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Log1p"));
  return __op->call(x);
}

Maybe<one::Tensor> Log1pGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log1pGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> LogSigmoid(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("LogSigmoid"));
  return __op->call(x);
}

Maybe<one::Tensor> LogSigmoidGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LogSigmoidGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Rint(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Rint"));
  return __op->call(x);
}

Maybe<one::Tensor> RintGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("RintGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Round(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Round"));
  return __op->call(x);
}

Maybe<one::Tensor> RoundGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("RoundGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Sign(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sign"));
  return __op->call(x);
}

Maybe<one::Tensor> SignGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SignGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Sinh(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Sinh"));
  return __op->call(x);
}

Maybe<one::Tensor> SinhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SinhGrad"));
  return __op->call(x, dy);
}

Maybe<one::Tensor> Softplus(const std::shared_ptr<one::Tensor>& x, double beta, double threshold) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, double>("Softplus"));
  return __op->call(x, beta, threshold);
}

Maybe<one::Tensor> SoftplusGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double beta, double threshold) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double>("SoftplusGrad"));
  return __op->call(x, dy, beta, threshold);
}

Maybe<one::Tensor> SoftShrink(const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, double, bool>("SoftShrink"));
  return __op->call(x, alpha, inplace);
}

Maybe<one::Tensor> SoftShrinkGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("SoftShrinkGrad"));
  return __op->call(y, dy, alpha);
}

Maybe<one::Tensor> OneHot(const std::shared_ptr<one::Tensor>& input, int64_t num_classes, const Scalar& on_value, const Scalar& off_value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Scalar&, const Scalar&>("OneHot"));
  return __op->call(input, num_classes, on_value, off_value);
}

Maybe<one::Tensor> UnsortedSegmentSumLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& segment_ids, const std::shared_ptr<one::Tensor>& like, int64_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("UnsortedSegmentSumLike"));
  return __op->call(x, segment_ids, like, axis);
}

Maybe<one::Tensor> UnsortedSegmentSum(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& segment_ids, int64_t axis, int64_t num_segments) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, int64_t>("UnsortedSegmentSum"));
  return __op->call(x, segment_ids, axis, num_segments);
}

Maybe<one::Tensor> Tril(const std::shared_ptr<one::Tensor>& x, int64_t diagonal) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t>("Tril"));
  return __op->call(x, diagonal);
}

Maybe<one::Tensor> Triu(const std::shared_ptr<one::Tensor>& x, int64_t diagonal) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t>("Triu"));
  return __op->call(x, diagonal);
}

Maybe<one::Tensor> InplaceTriu(const std::shared_ptr<one::Tensor>& x, int64_t diagonal) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t>("InplaceTriu"));
  return __op->call(x, diagonal);
}

Maybe<one::Tensor> Clamp(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<Scalar>&>("Clamp"));
  return __op->call(input, min, max);
}

Maybe<one::Tensor> ClampInplace(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<Scalar>&>("ClampInplace"));
  return __op->call(input, min, max);
}

Maybe<one::Tensor> ClampMin(const std::shared_ptr<one::Tensor>& input, const Scalar& min) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ClampMin"));
  return __op->call(input, min);
}

Maybe<one::Tensor> ClampMinInplace(const std::shared_ptr<one::Tensor>& input, const Scalar& min) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ClampMinInplace"));
  return __op->call(input, min);
}

Maybe<one::Tensor> ClampMax(const std::shared_ptr<one::Tensor>& input, const Scalar& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ClampMax"));
  return __op->call(input, max);
}

Maybe<one::Tensor> ClampMaxInplace(const std::shared_ptr<one::Tensor>& input, const Scalar& min) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("ClampMaxInplace"));
  return __op->call(input, min);
}

Maybe<one::Tensor> Clip(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<Scalar>&>("Clip"));
  return __op->call(input, min, max);
}

Maybe<one::Tensor> ClipInplace(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<Scalar>&>("ClipInplace"));
  return __op->call(input, min, max);
}

Maybe<one::Tensor> ClampGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min, const Optional<Scalar>& max) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<Scalar>&>("ClampGrad"));
  return __op->call(dy, x, min, max);
}

Maybe<one::Tensor> VectorNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const Optional<std::vector<int32_t>>&, bool, const Optional<Symbol<DType>>&>("VectorNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> VectorNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const Scalar&, bool, const Optional<Symbol<DType>>&>("VectorNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> MatrixNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&, const std::vector<int32_t>&, bool, const Optional<Symbol<DType>>&>("MatrixNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> MatrixNorm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&, const std::vector<int32_t>&, bool, const Optional<Symbol<DType>>&>("MatrixNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> Norm(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype, bool for_norm) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Optional<std::vector<int32_t>>&, bool, const Optional<Symbol<DType>>&, bool>("Norm"));
  return __op->call(input, ord, dim, keepdim, dtype, for_norm);
}

Maybe<one::Tensor> Norm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&, const Optional<std::vector<int32_t>>&, bool, const Optional<Symbol<DType>>&>("Norm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> ScalarNorm(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Scalar>&, const Scalar&, bool, const Optional<Symbol<DType>>&>("ScalarNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> ScalarNorm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&, const Scalar&, bool, const Optional<Symbol<DType>>&>("ScalarNorm"));
  return __op->call(input, ord, dim, keepdim, dtype);
}

Maybe<one::Tensor> Inv(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Inv"));
  return __op->call(x);
}

Maybe<one::Tensor> LinalgCross(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Optional<int64_t>& dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<int64_t>&>("LinalgCross"));
  return __op->call(input, other, dim);
}

Maybe<one::Tensor> Dropout(const std::shared_ptr<one::Tensor>& input, float p, bool training, bool inplace, const Optional<one::Generator>& generator, const Optional<one::Tensor>& addend) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, bool, bool, const Optional<one::Generator>&, const Optional<one::Tensor>&>("Dropout"));
  return __op->call(input, p, training, inplace, generator, addend);
}

Maybe<one::Tensor> DropoutGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("DropoutGrad"));
  return __op->call(dy, mask, scale);
}

Maybe<one::Tensor> Dropout1d(const std::shared_ptr<one::Tensor>& input, float p, bool training) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, bool>("Dropout1d"));
  return __op->call(input, p, training);
}

Maybe<one::Tensor> Dropout2d(const std::shared_ptr<one::Tensor>& input, float p, bool training) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, bool>("Dropout2d"));
  return __op->call(input, p, training);
}

Maybe<one::Tensor> Dropout3d(const std::shared_ptr<one::Tensor>& input, float p, bool training) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, bool>("Dropout3d"));
  return __op->call(input, p, training);
}

Maybe<one::Tensor> ConstantPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, const Scalar&>("ConstantPad"));
  return __op->call(x, pad, value);
}

Maybe<one::Tensor> ReflectionPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("ReflectionPad"));
  return __op->call(x, pad);
}

Maybe<one::Tensor> ReplicationPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("ReplicationPad"));
  return __op->call(x, pad);
}

Maybe<one::Tensor> Pad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad, const std::string& mode, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, const std::string&, const Scalar&>("Pad"));
  return __op->call(x, pad, mode, value);
}

Maybe<one::Tensor> PadGrad(const std::shared_ptr<one::Tensor>& dy, const std::vector<int64_t>& pad, const std::string& mode, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, const std::string&, const Scalar&>("PadGrad"));
  return __op->call(dy, pad, mode, value);
}

Maybe<one::Tensor> Silu(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Silu"));
  return __op->call(x);
}

Maybe<one::Tensor> SiluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SiluGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> Mish(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Mish"));
  return __op->call(x);
}

Maybe<one::Tensor> MishGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MishGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> Selu(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Selu"));
  return __op->call(x);
}

Maybe<one::Tensor> SeluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SeluGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> SoftSign(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("SoftSign"));
  return __op->call(x);
}

Maybe<one::Tensor> SoftSignGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SoftSignGrad"));
  return __op->call(dy, x);
}

Maybe<one::Tensor> Diag(const std::shared_ptr<one::Tensor>& x, int32_t diagonal) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("Diag"));
  return __op->call(x, diagonal);
}

Maybe<one::Tensor> DiagGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& in, int32_t diagonal) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("DiagGrad"));
  return __op->call(dy, in, diagonal);
}

Maybe<one::Tensor> Diagonal(const std::shared_ptr<one::Tensor>& x, int32_t offset, int32_t dim1, int32_t dim2) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t, int32_t>("Diagonal"));
  return __op->call(x, offset, dim1, dim2);
}

Maybe<one::Tensor> DiagonalGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& in, int32_t offset) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("DiagonalGrad"));
  return __op->call(dy, in, offset);
}

Maybe<one::Tensor> TensorGetItem(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const TensorIndex&>("TensorGetItem"));
  return __op->call(x, index);
}

Maybe<one::Tensor> DimScatter(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, const Optional<std::string>& reduce, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<std::string>&, bool>("DimScatter"));
  return __op->call(input, dim, index, src, reduce, inplace);
}

Maybe<one::Tensor> DimScatterScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, const Optional<std::string>& reduce, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const Scalar&, const Optional<std::string>&, bool>("DimScatterScalar"));
  return __op->call(input, dim, index, src, reduce, inplace);
}

Maybe<one::Tensor> DimScatterUpdate(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("DimScatterUpdate"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterUpdateScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("DimScatterUpdateScalar"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterAdd(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("DimScatterAdd"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterAddScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("DimScatterAddScalar"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterMul(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("DimScatterMul"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterMulScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const Scalar&, bool>("DimScatterMulScalar"));
  return __op->call(input, dim, index, src, inplace);
}

Maybe<one::Tensor> DimScatterAddLike(const std::shared_ptr<one::Tensor>& like, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("DimScatterAddLike"));
  return __op->call(like, dim, index, src);
}

Maybe<void> TensorSetItem(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index, const std::shared_ptr<one::Tensor>& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const std::shared_ptr<one::Tensor>&, const TensorIndex&, const std::shared_ptr<one::Tensor>&>("TensorSetItem"));
  return __op->call(x, index, value);
}

Maybe<one::Tensor> AvgPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, bool, bool, int32_t, const std::string&>("AvgPool1D"));
  return __op->call(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, data_format);
}

Maybe<one::Tensor> AvgPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, bool, bool, int32_t, const std::string&>("AvgPool2D"));
  return __op->call(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, data_format);
}

Maybe<one::Tensor> AvgPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const Optional<std::vector<int32_t>>&, const std::vector<int32_t>&, bool, bool, int32_t, const std::string&>("AvgPool3D"));
  return __op->call(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, data_format);
}

Maybe<one::Tensor> AvgPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, int32_t ndims, const std::string& data_format, const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, bool ceil_mode, bool count_include_pad, int32_t divisor_override) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, const std::string&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, bool, bool, int32_t>("AvgPoolNdGrad"));
  return __op->call(x, dy, ndims, data_format, padding, kernel_size, stride, ceil_mode, count_include_pad, divisor_override);
}

Maybe<one::Tensor> Minimum(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Minimum"));
  return __op->call(input, other);
}

Maybe<one::Tensor> Maximum(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Maximum"));
  return __op->call(input, other);
}

Maybe<one::TensorTuple> ElementwiseMinGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ElementwiseMinGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::TensorTuple> ElementwiseMaxGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ElementwiseMaxGrad"));
  return __op->call(dz, x, y);
}

Maybe<one::Tensor> Stack(const TensorTuple& inputs, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&, int64_t>("Stack"));
  return __op->call(inputs, dim);
}

Maybe<one::TensorTuple> StackGrad(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like, int64_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, int64_t>("StackGrad"));
  return __op->call(x, like, axis);
}

Maybe<one::Tensor> AtLeast1D(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("AtLeast1D"));
  return __op->call(input);
}

Maybe<one::TensorTuple> AtLeast1D(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&>("AtLeast1D"));
  return __op->call(tensors);
}

Maybe<one::Tensor> AtLeast2D(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("AtLeast2D"));
  return __op->call(input);
}

Maybe<one::TensorTuple> AtLeast2D(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&>("AtLeast2D"));
  return __op->call(tensors);
}

Maybe<one::Tensor> AtLeast3D(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("AtLeast3D"));
  return __op->call(input);
}

Maybe<one::TensorTuple> AtLeast3D(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&>("AtLeast3D"));
  return __op->call(tensors);
}

Maybe<one::Tensor> HStack(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&>("HStack"));
  return __op->call(tensors);
}

Maybe<one::Tensor> VStack(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&>("VStack"));
  return __op->call(tensors);
}

Maybe<one::Tensor> DStack(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&>("DStack"));
  return __op->call(tensors);
}

Maybe<one::Tensor> ColumnStack(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&>("ColumnStack"));
  return __op->call(tensors);
}

Maybe<one::Tensor> RowStack(const TensorTuple& tensors) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&>("RowStack"));
  return __op->call(tensors);
}

Maybe<one::Tensor> LocalToGlobal(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Shape& shape, const Symbol<DType>& dtype, bool sync_data, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Shape&, const Symbol<DType>&, bool, bool>("LocalToGlobal"));
  return __op->call(x, placement, sbp, shape, dtype, sync_data, copy);
}

Maybe<one::Tensor> ToGlobal(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const std::vector<Symbol<SbpParallel>>& grad_sbp, bool check_meta, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const std::vector<Symbol<SbpParallel>>&, bool, bool>("ToGlobal"));
  return __op->call(x, placement, sbp, grad_sbp, check_meta, copy);
}

Maybe<one::Tensor> GlobalToLocal(const std::shared_ptr<one::Tensor>& x, bool copy) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, bool>("GlobalToLocal"));
  return __op->call(x, copy);
}

Maybe<void> StreamTouch(const TensorTuple& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const TensorTuple&>("StreamTouch"));
  return __op->call(x);
}

Maybe<one::TensorTuple> BroadcastTensors(const TensorTuple& inputs, int64_t src_rank, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&, int64_t, bool>("BroadcastTensors"));
  return __op->call(inputs, src_rank, inplace);
}

Maybe<one::Tensor> LocalAllReduce(const std::shared_ptr<one::Tensor>& x, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, bool>("LocalAllReduce"));
  return __op->call(x, inplace);
}

Maybe<one::Tensor> LocalReduce(const std::shared_ptr<one::Tensor>& x, int64_t dst, bool inplace) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, bool>("LocalReduce"));
  return __op->call(x, dst, inplace);
}

Maybe<one::Tensor> EagerPToB(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const Shape&>("EagerPToB"));
  return __op->call(x, in_placement, out_placement, shape);
}

Maybe<one::Tensor> EagerBToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Shape&>("EagerBToS"));
  return __op->call(x, in_placement, out_placement, out_sbp, shape);
}

Maybe<one::Tensor> EagerSToB(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& in_sbp, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Shape&>("EagerSToB"));
  return __op->call(x, in_placement, out_placement, in_sbp, shape);
}

Maybe<one::Tensor> EagerNaiveSToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& in_sbp, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const std::vector<Symbol<SbpParallel>>&, const Shape&>("EagerNaiveSToS"));
  return __op->call(x, in_placement, out_placement, in_sbp, out_sbp, shape);
}

Maybe<one::Tensor> EagerPToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Shape&>("EagerPToS"));
  return __op->call(x, in_placement, out_placement, out_sbp, shape);
}

Maybe<one::Tensor> EagerSToP(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Shape&>("EagerSToP"));
  return __op->call(x, in_placement, out_placement, out_sbp, shape);
}

Maybe<one::Tensor> GlobalAllReduce(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("GlobalAllReduce"));
  return __op->call(x);
}

Maybe<one::Tensor> GlobalReduceScatter(const std::shared_ptr<one::Tensor>& x, const std::string& op_type) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&>("GlobalReduceScatter"));
  return __op->call(x, op_type);
}

Maybe<one::Tensor> GlobalAllGather(const std::shared_ptr<one::Tensor>& x) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("GlobalAllGather"));
  return __op->call(x);
}

Maybe<one::Tensor> GlobalS2S(const std::shared_ptr<one::Tensor>& x, const std::vector<Symbol<SbpParallel>>& out_sbp) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<Symbol<SbpParallel>>&>("GlobalS2S"));
  return __op->call(x, out_sbp);
}

Maybe<one::TensorTuple> SelectTopN(const TensorTuple& inputs, int32_t n) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&, int32_t>("SelectTopN"));
  return __op->call(inputs, n);
}

Maybe<one::Tensor> CastLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& like) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CastLike"));
  return __op->call(x, like);
}

Maybe<one::Tensor> Identity(const std::shared_ptr<one::Tensor>& in) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Identity"));
  return __op->call(in);
}

Maybe<one::Tensor> AmpWhiteIdentity(const std::shared_ptr<one::Tensor>& in) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("AmpWhiteIdentity"));
  return __op->call(in);
}

Maybe<one::Tensor> AmpBlackIdentity(const std::shared_ptr<one::Tensor>& in) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("AmpBlackIdentity"));
  return __op->call(in);
}

Maybe<one::Tensor> ReshapeLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReshapeLike"));
  return __op->call(in, like);
}

Maybe<one::Tensor> ReduceSumLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&>("ReduceSumLike"));
  return __op->call(in, like, axis);
}

Maybe<one::Tensor> BroadcastReduceSumLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BroadcastReduceSumLike"));
  return __op->call(in, like);
}

Maybe<one::Tensor> Rand(const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("Rand"));
  return __op->call(size, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRand(const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRand"));
  return __op->call(size, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> RandN(const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandN"));
  return __op->call(size, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandN(const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandN"));
  return __op->call(size, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> RandnLike(const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandnLike"));
  return __op->call(input, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandnLike(const std::shared_ptr<one::Tensor>& input, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandnLike"));
  return __op->call(input, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> RandInt(int64_t low, int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, int64_t, const Shape&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandInt"));
  return __op->call(low, high, size, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> RandInt(int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, const Shape&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandInt"));
  return __op->call(high, size, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandInt(int64_t low, int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, int64_t, const Shape&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandInt"));
  return __op->call(low, high, size, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandInt(int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, const Shape&, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandInt"));
  return __op->call(high, size, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> RandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandIntLike"));
  return __op->call(x, low, high, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> RandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Generator>&, bool>("RandIntLike"));
  return __op->call(x, high, dtype, device, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandIntLike"));
  return __op->call(x, low, high, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> GlobalRandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<Symbol<DType>>&, const Optional<one::Generator>&, bool>("GlobalRandIntLike"));
  return __op->call(x, high, placement, sbp, dtype, generator, requires_grad);
}

Maybe<one::Tensor> RandPerm(int32_t n, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int32_t, const Optional<one::Generator>&, const Symbol<DType>&, const Optional<Symbol<Device>>&, bool>("RandPerm"));
  return __op->call(n, generator, dtype, device, requires_grad);
}

Maybe<one::Tensor> GlobalRandPerm(int32_t n, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, bool requires_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int32_t, const Symbol<ParallelDesc>&, const std::vector<Symbol<SbpParallel>>&, const Optional<one::Generator>&, const Symbol<DType>&, bool>("GlobalRandPerm"));
  return __op->call(n, placement, sbp, generator, dtype, requires_grad);
}

Maybe<one::Tensor> UnfoldTensor(const std::shared_ptr<one::Tensor>& x, int32_t dimension, int32_t size, int32_t step) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, int32_t, int32_t>("UnfoldTensor"));
  return __op->call(x, dimension, size, step);
}

Maybe<one::Tensor> UnfoldTensorGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, int32_t dimension, int32_t size, int32_t step) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, int32_t, int32_t>("UnfoldTensorGrad"));
  return __op->call(dy, x, dimension, size, step);
}

Maybe<one::Tensor> Unfold(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::string&>("Unfold"));
  return __op->call(x, kernel_size, dilation, padding, stride, data_format);
}

Maybe<one::Tensor> Fold(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& output_size, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::string&>("Fold"));
  return __op->call(x, output_size, kernel_size, dilation, padding, stride, data_format);
}

Maybe<one::TensorTuple> Split(const std::shared_ptr<one::Tensor>& x, int64_t split_size_or_sections, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t>("Split"));
  return __op->call(x, split_size_or_sections, dim);
}

Maybe<one::TensorTuple> SplitWithSize(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& split_size_or_sections, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&, int64_t>("SplitWithSize"));
  return __op->call(x, split_size_or_sections, dim);
}

Maybe<one::TensorTuple> Unbind(const std::shared_ptr<one::Tensor>& x, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int64_t>("Unbind"));
  return __op->call(x, dim);
}

Maybe<one::TensorTuple> Chunk(const std::shared_ptr<one::Tensor>& x, int64_t chunks, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t>("Chunk"));
  return __op->call(x, chunks, dim);
}

Maybe<one::TensorTuple> SplitLike(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like, int64_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, int64_t>("SplitLike"));
  return __op->call(x, like, axis);
}

Maybe<one::Tensor> PairwiseDistance(const std::shared_ptr<one::Tensor>& x1, const std::shared_ptr<one::Tensor>& x2, float p, double eps, bool keepdim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, double, bool>("PairwiseDistance"));
  return __op->call(x1, x2, p, eps, keepdim);
}

Maybe<one::Tensor> CosineSimilarity(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, int32_t dim, double eps) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, double>("CosineSimilarity"));
  return __op->call(x, y, dim, eps);
}

Maybe<one::Tensor> Normalize(const std::shared_ptr<one::Tensor>& input, float p, int32_t dim, float eps, bool use_l2_norm_kernel) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, int32_t, float, bool>("Normalize"));
  return __op->call(input, p, dim, eps, use_l2_norm_kernel);
}

Maybe<one::Tensor> L2Normalize(const std::shared_ptr<one::Tensor>& input, int32_t axis, float epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, float>("L2Normalize"));
  return __op->call(input, axis, epsilon);
}

Maybe<one::Tensor> L2NormalizeGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& square_x_sum, int32_t axis, float epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, float>("L2NormalizeGrad"));
  return __op->call(dy, y, square_x_sum, axis, epsilon);
}

Maybe<one::TensorTuple> FusedSelfAttention(const std::shared_ptr<one::Tensor>& hidden_states, int64_t head_size, float alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int64_t, float>("FusedSelfAttention"));
  return __op->call(hidden_states, head_size, alpha);
}

Maybe<one::Tensor> FusedSelfAttentionGrad(const std::shared_ptr<one::Tensor>& query_mul_key_grad, const std::shared_ptr<one::Tensor>& value_grad, const std::shared_ptr<one::Tensor>& hidden_states, float alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("FusedSelfAttentionGrad"));
  return __op->call(query_mul_key_grad, value_grad, hidden_states, alpha);
}

Maybe<one::Tensor> FusedScaleTril(const std::shared_ptr<one::Tensor>& x, int64_t diagonal, const Scalar& fill_value, const Scalar& scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Scalar&, const Scalar&>("FusedScaleTril"));
  return __op->call(x, diagonal, fill_value, scale);
}

Maybe<one::Tensor> FusedBiasAddGelu(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("FusedBiasAddGelu"));
  return __op->call(a, b, axis);
}

Maybe<one::Tensor> FusedBiasAddGeluGrad(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, const std::shared_ptr<one::Tensor>& dy, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("FusedBiasAddGeluGrad"));
  return __op->call(a, b, dy, axis);
}

Maybe<one::Tensor> FusedBiasAddDropout(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, float p, int32_t axis, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, int32_t, const Optional<one::Generator>&>("FusedBiasAddDropout"));
  return __op->call(a, b, p, axis, generator);
}

Maybe<one::Tensor> FusedScaleMaskSoftmax(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float>("FusedScaleMaskSoftmax"));
  return __op->call(x, mask, fill_value, scale);
}

Maybe<one::Tensor> FusedScaleMaskSoftmaxGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float>("FusedScaleMaskSoftmaxGrad"));
  return __op->call(y, dy, mask, scale);
}

Maybe<one::TensorTuple> FusedScaleMaskSoftmaxDropout(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale, float p, bool training, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float, float, bool, const Optional<one::Generator>&>("FusedScaleMaskSoftmaxDropout"));
  return __op->call(x, mask, fill_value, scale, p, training, generator);
}

Maybe<one::Tensor> FusedScaleMaskSoftmaxDropoutGrad(const std::shared_ptr<one::Tensor>& softmax_y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& dropout_mask, float scale, float dropout_scale) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, float>("FusedScaleMaskSoftmaxDropoutGrad"));
  return __op->call(softmax_y, dy, mask, dropout_mask, scale, dropout_scale);
}

Maybe<one::TensorTuple> FusedScaleTrilSoftmaxMaskScale(const std::shared_ptr<one::Tensor>& a, float p, int64_t diagonal, float tril_scale_value, float tril_fill_value, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, float, int64_t, float, float, const Optional<one::Generator>&>("FusedScaleTrilSoftmaxMaskScale"));
  return __op->call(a, p, diagonal, tril_scale_value, tril_fill_value, generator);
}

Maybe<one::Tensor> FusedScaleTrilSoftmaxMaskScaleGrad(const std::shared_ptr<one::Tensor>& softmax_y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, int64_t diagonal, float tril_scale_value, float mask_scale_value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, float, float>("FusedScaleTrilSoftmaxMaskScaleGrad"));
  return __op->call(softmax_y, dy, mask, diagonal, tril_scale_value, mask_scale_value);
}

Maybe<one::Tensor> FusedMultiHeadAttentionInference(const std::shared_ptr<one::Tensor>& query, const std::shared_ptr<one::Tensor>& key, const std::shared_ptr<one::Tensor>& value, int64_t num_heads, bool causal, int64_t query_hidden_slice_start, int64_t query_hidden_slice_end, int64_t key_hidden_slice_start, int64_t key_hidden_slice_end, int64_t value_hidden_slice_start, int64_t value_hidden_slice_end) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>("FusedMultiHeadAttentionInference"));
  return __op->call(query, key, value, num_heads, causal, query_hidden_slice_start, query_hidden_slice_end, key_hidden_slice_start, key_hidden_slice_end, value_hidden_slice_start, value_hidden_slice_end);
}

Maybe<void> Send(const std::shared_ptr<one::Tensor>& input, int64_t dst, bool send_meta) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const std::shared_ptr<one::Tensor>&, int64_t, bool>("Send"));
  return __op->call(input, dst, send_meta);
}

Maybe<one::Tensor> Recv(int64_t src, const Optional<Shape>& shape, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Tensor>& out) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, int64_t, const Optional<Shape>&, const Optional<Symbol<DType>>&, const Optional<Symbol<Device>>&, const Optional<one::Tensor>&>("Recv"));
  return __op->call(src, shape, dtype, device, out);
}

Maybe<one::Tensor> BatchGather(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("BatchGather"));
  return __op->call(in, indices);
}

Maybe<one::Tensor> UnsortedBatchSegmentSum(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& segment_ids, int64_t num_segments) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("UnsortedBatchSegmentSum"));
  return __op->call(data, segment_ids, num_segments);
}

Maybe<one::TensorTuple> CtcGreedyDecoder(const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& input_lengths, bool merge_repeated) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("CtcGreedyDecoder"));
  return __op->call(log_probs, input_lengths, merge_repeated);
}

Maybe<one::TensorTuple> DistributedPariticalFCSampleDisableBoxing(const std::shared_ptr<one::Tensor>& sampled_weight_diff, const std::shared_ptr<one::Tensor>& sampled_label) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("DistributedPariticalFCSampleDisableBoxing"));
  return __op->call(sampled_weight_diff, sampled_label);
}

Maybe<one::Tensor> Nms(const std::shared_ptr<one::Tensor>& x, float iou_threshold, int32_t keep_n) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, int32_t>("Nms"));
  return __op->call(x, iou_threshold, keep_n);
}

Maybe<one::Tensor> RoiAlign(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& rois, float spatial_scale, int32_t pooled_h, int32_t pooled_w, int32_t sampling_ratio, bool aligned) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, int32_t, int32_t, int32_t, bool>("RoiAlign"));
  return __op->call(x, rois, spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned);
}

Maybe<one::Tensor> RoiAlignGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x_like, const std::shared_ptr<one::Tensor>& rois, float spatial_scale, int32_t pooled_h, int32_t pooled_w, int32_t sampling_ratio, bool aligned) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, float, int32_t, int32_t, int32_t, bool>("RoiAlignGrad"));
  return __op->call(dy, x_like, rois, spatial_scale, pooled_h, pooled_w, sampling_ratio, aligned);
}

Maybe<one::TensorTuple> Meshgrid(const TensorTuple& tensors, const std::string& indexing) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const TensorTuple&, const std::string&>("Meshgrid"));
  return __op->call(tensors, indexing);
}

Maybe<one::Tensor> IndexSelect(const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const std::shared_ptr<one::Tensor>&>("IndexSelect"));
  return __op->call(input, dim, index);
}

Maybe<one::Tensor> DecodeOneRec(const std::shared_ptr<one::Tensor>& input, const std::string& key, const Symbol<DType>& dtype, const Shape& shape, bool is_dynamic, const Optional<Shape>& reshape, const Optional<Shape>& batch_padding) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::string&, const Symbol<DType>&, const Shape&, bool, const Optional<Shape>&, const Optional<Shape>&>("DecodeOneRec"));
  return __op->call(input, key, dtype, shape, is_dynamic, reshape, batch_padding);
}

Maybe<one::Tensor> Dot(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Dot"));
  return __op->call(input, other);
}

Maybe<one::Tensor> FusedDotFeatureInteraction(const TensorTuple& features, const Optional<one::Tensor>& output_concat, bool self_interaction, int32_t output_padding, const std::string& pooling) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const TensorTuple&, const Optional<one::Tensor>&, bool, int32_t, const std::string&>("FusedDotFeatureInteraction"));
  return __op->call(features, output_concat, self_interaction, output_padding, pooling);
}

Maybe<one::TensorTuple> FusedDotFeatureInteractionGrad(const std::shared_ptr<one::Tensor>& dy, const TensorTuple& features, bool has_output_concat_grad, bool self_interaction, int32_t output_concat_grad_dim, const std::string& pooling) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, bool, int32_t, const std::string&>("FusedDotFeatureInteractionGrad"));
  return __op->call(dy, features, has_output_concat_grad, self_interaction, output_concat_grad_dim, pooling);
}

Maybe<one::Tensor> FusedCrossFeatureInteraction(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& bias, const std::string& interaction_mode) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&>("FusedCrossFeatureInteraction"));
  return __op->call(x, weight, x_0, bias, interaction_mode);
}

Maybe<one::TensorTuple> FusedCrossFeatureInteractionV1Grad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& matmul_result) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FusedCrossFeatureInteractionV1Grad"));
  return __op->call(dy, weight, x, x_0, matmul_result);
}

Maybe<one::TensorTuple> FusedCrossFeatureInteractionV2Grad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& bias, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& matmul_result) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FusedCrossFeatureInteractionV2Grad"));
  return __op->call(dy, weight, bias, x, x_0, matmul_result);
}

Maybe<one::Tensor> TensorBufferToTensor(const std::shared_ptr<one::Tensor>& input, const Shape& instance_shape, const Symbol<DType>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Shape&, const Symbol<DType>&>("TensorBufferToTensor"));
  return __op->call(input, instance_shape, dtype);
}

Maybe<one::Tensor> TensorToTensorBuffer(const std::shared_ptr<one::Tensor>& input, int32_t instance_dims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("TensorToTensorBuffer"));
  return __op->call(input, instance_dims);
}

Maybe<one::Tensor> GenTensorBuffer(const Shape& shape, const std::vector<Shape>& shape_list, const std::vector<float>& value_list, const Symbol<DType>& data_type, bool dynamic_out) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const Shape&, const std::vector<Shape>&, const std::vector<float>&, const Symbol<DType>&, bool>("GenTensorBuffer"));
  return __op->call(shape, shape_list, value_list, data_type, dynamic_out);
}

Maybe<one::Tensor> TopK(const std::shared_ptr<one::Tensor>& input, int32_t k, bool sorted) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, bool>("TopK"));
  return __op->call(input, k, sorted);
}

Maybe<one::Tensor> InTopK(const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& predictions, int32_t k) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("InTopK"));
  return __op->call(targets, predictions, k);
}

Maybe<one::Tensor> Cumsum(const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Optional<Symbol<DType>>&>("Cumsum"));
  return __op->call(input, dim, dtype);
}

Maybe<one::Tensor> Cumprod(const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, const Optional<Symbol<DType>>&>("Cumprod"));
  return __op->call(input, dim, dtype);
}

Maybe<one::Tensor> CumprodGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& x, int64_t dim) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int64_t>("CumprodGrad"));
  return __op->call(input, y, x, dim);
}

Maybe<one::TensorTuple> OneEmbeddingIdShuffle(const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, int32_t num_tables, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int32_t, const std::string&>("OneEmbeddingIdShuffle"));
  return __op->call(ids, table_ids, num_tables, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingEmbeddingShuffle(const std::shared_ptr<one::Tensor>& cur_rank_embeddings, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&>("OneEmbeddingEmbeddingShuffle"));
  return __op->call(cur_rank_embeddings, num_unique_matrix, cur_rank_inverse_indices, inverse_unique_partition_indices, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingEmbeddingGradientShuffle(const std::shared_ptr<one::Tensor>& embedding_grad, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&>("OneEmbeddingEmbeddingGradientShuffle"));
  return __op->call(embedding_grad, num_unique_matrix, cur_rank_inverse_indices, inverse_unique_partition_indices, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingLookup(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& table_ids, const Symbol<DType>& dtype, const Symbol<DType>& embedding_dtype, int64_t line_size, int64_t embedding_size, const std::string& embedding_name, const std::string& embedding_tables, const std::string& state_initializer, int64_t seed) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Symbol<DType>&, const Symbol<DType>&, int64_t, int64_t, const std::string&, const std::string&, const std::string&, int64_t>("OneEmbeddingLookup"));
  return __op->call(num_unique_ids, unique_ids, table_ids, dtype, embedding_dtype, line_size, embedding_size, embedding_name, embedding_tables, state_initializer, seed);
}

Maybe<one::Tensor> OneEmbeddingFusedLookup(const std::shared_ptr<one::Tensor>& shadow, const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype, const std::string& embedding_name, int64_t line_size, int64_t embedding_size, bool is_full_cache, int32_t num_tables, const std::string& embedding_tables, const Optional<int64_t>& padding_idx, int64_t seed) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Symbol<DType>&, const std::string&, int64_t, int64_t, bool, int32_t, const std::string&, const Optional<int64_t>&, int64_t>("OneEmbeddingFusedLookup"));
  return __op->call(shadow, ids, table_ids, dtype, embedding_name, line_size, embedding_size, is_full_cache, num_tables, embedding_tables, padding_idx, seed);
}

Maybe<void> OneEmbeddingFusedLookupGrad(const std::shared_ptr<one::Tensor>& ids, const std::shared_ptr<one::Tensor>& embedding_grad, const std::string& embedding_name, int64_t line_size, int64_t embedding_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int64_t, int64_t>("OneEmbeddingFusedLookupGrad"));
  return __op->call(ids, embedding_grad, embedding_name, line_size, embedding_size);
}

Maybe<one::TensorTuple> OneEmbeddingUniqueKeyValuePair(const std::shared_ptr<one::Tensor>& keys, const Optional<one::Tensor>& values, int32_t num_tables, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int32_t, const std::string&>("OneEmbeddingUniqueKeyValuePair"));
  return __op->call(keys, values, num_tables, embedding_name);
}

Maybe<void> OneEmbeddingEmbeddingPut(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::string& embedding_name, int64_t line_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::string&, int64_t>("OneEmbeddingEmbeddingPut"));
  return __op->call(num_unique_ids, unique_ids, unique_embeddings, embedding_name, line_size);
}

Maybe<one::Tensor> OneEmbeddingSgdUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float momentum, int64_t line_size, int64_t embedding_size, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, float, double, float, float, int64_t, int64_t, const std::string&>("OneEmbeddingSgdUpdate"));
  return __op->call(num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor, skip_if, learning_rate_val, scale, weight_decay, momentum, line_size, embedding_size, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingAdamUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& bias_correction1, const Optional<one::Tensor>& bias_correction2, float learning_rate_val, double scale, float weight_decay, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, float epsilon, bool do_bias_correction, int64_t line_size, int64_t embedding_size, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, float, double, float, float, float, float, float, float, bool, int64_t, int64_t, const std::string&>("OneEmbeddingAdamUpdate"));
  return __op->call(num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor, skip_if, bias_correction1, bias_correction2, learning_rate_val, scale, weight_decay, beta1, beta2, bias_correction1_val, bias_correction2_val, epsilon, do_bias_correction, line_size, embedding_size, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingAdagradUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& train_step, int64_t train_step_val, float learning_rate_val, double scale, float weight_decay, float lr_decay, float epsilon, int64_t line_size, int64_t embedding_size, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, int64_t, float, double, float, float, float, int64_t, int64_t, const std::string&>("OneEmbeddingAdagradUpdate"));
  return __op->call(num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor, skip_if, train_step, train_step_val, learning_rate_val, scale, weight_decay, lr_decay, epsilon, line_size, embedding_size, embedding_name);
}

Maybe<one::Tensor> OneEmbeddingFtrlUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float lr_power, float lambda1, float lambda2, float beta, int64_t line_size, int64_t embedding_size, const std::string& embedding_name) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, float, double, float, float, float, float, float, int64_t, int64_t, const std::string&>("OneEmbeddingFtrlUpdate"));
  return __op->call(num_unique_ids, unique_embeddings, embedding_grad, learning_rate, down_scale_by_tensor, skip_if, learning_rate_val, scale, weight_decay, lr_power, lambda1, lambda2, beta, line_size, embedding_size, embedding_name);
}

Maybe<one::Tensor> EinSum(const std::string& equation, const TensorTuple& operands) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::string&, const TensorTuple&>("EinSum"));
  return __op->call(equation, operands);
}

Maybe<one::Tensor> PixelShuffle(const std::shared_ptr<one::Tensor>& input, int64_t h_upscale_factor, int64_t w_upscale_factor) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int64_t, int64_t>("PixelShuffle"));
  return __op->call(input, h_upscale_factor, w_upscale_factor);
}

Maybe<one::Tensor> IsNan(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("IsNan"));
  return __op->call(input);
}

Maybe<one::Tensor> IsInf(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("IsInf"));
  return __op->call(input);
}

Maybe<one::Tensor> IsFinite(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("IsFinite"));
  return __op->call(input);
}

Maybe<one::Tensor> RocAucScore(const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& pred) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("RocAucScore"));
  return __op->call(label, pred);
}

Maybe<one::Tensor> PinMemory(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("PinMemory"));
  return __op->call(input);
}

Maybe<one::Tensor> FillTensor(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("FillTensor"));
  return __op->call(in, value);
}

Maybe<one::Tensor> Fill(const std::shared_ptr<one::Tensor>& in, const Scalar& value) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Scalar&>("Fill"));
  return __op->call(in, value);
}

Maybe<one::Tensor> RnnTanhCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("RnnTanhCell"));
  return __op->call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

Maybe<one::Tensor> RnnReluCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("RnnReluCell"));
  return __op->call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

Maybe<one::TensorTuple> LstmCell(const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("LstmCell"));
  return __op->call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

Maybe<one::Tensor> GruCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("GruCell"));
  return __op->call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

Maybe<one::TensorTuple> FusedGruCell(const std::shared_ptr<one::Tensor>& igates, const std::shared_ptr<one::Tensor>& hgates, const std::shared_ptr<one::Tensor>& hx, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("FusedGruCell"));
  return __op->call(igates, hgates, hx, b_ih, b_hh);
}

Maybe<one::TensorTuple> FusedGruCellGrad(const std::shared_ptr<one::Tensor>& grad_hy, const std::shared_ptr<one::Tensor>& workspace, bool has_bias, bool hx_needs_grad) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, bool>("FusedGruCellGrad"));
  return __op->call(grad_hy, workspace, has_bias, hx_needs_grad);
}

Maybe<one::TensorTuple> FusedLstmCell(const std::shared_ptr<one::Tensor>& igates, const std::shared_ptr<one::Tensor>& hgates, const std::shared_ptr<one::Tensor>& cx, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&>("FusedLstmCell"));
  return __op->call(igates, hgates, cx, b_ih, b_hh);
}

Maybe<one::TensorTuple> FusedLstmCellGrad(const std::shared_ptr<one::Tensor>& grad_hy, const std::shared_ptr<one::Tensor>& grad_cy, const std::shared_ptr<one::Tensor>& cx, const std::shared_ptr<one::Tensor>& cy, const std::shared_ptr<one::Tensor>& workspace, bool need_cx_grad, bool has_bias) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool, bool>("FusedLstmCellGrad"));
  return __op->call(grad_hy, grad_cy, cx, cy, workspace, need_cx_grad, has_bias);
}

Maybe<one::TensorTuple> RnnTanhInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool, bool>("RnnTanhInput"));
  return __op->call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Maybe<one::TensorTuple> RnnTanhData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool>("RnnTanhData"));
  return __op->call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Maybe<one::TensorTuple> RnnReluInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool, bool>("RnnReluInput"));
  return __op->call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Maybe<one::TensorTuple> RnnReluData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool>("RnnReluData"));
  return __op->call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Maybe<one::TensorTuple> LstmInput(const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const TensorTuple&, bool, int32_t, float, bool, bool, bool>("LstmInput"));
  return __op->call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Maybe<one::TensorTuple> LstmData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, const TensorTuple&, bool, int32_t, float, bool, bool>("LstmData"));
  return __op->call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Maybe<one::TensorTuple> GruInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool, bool>("GruInput"));
  return __op->call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

Maybe<one::TensorTuple> GruData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const TensorTuple&, bool, int32_t, float, bool, bool>("GruData"));
  return __op->call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

Maybe<one::TensorTuple> PackPaddedSequence(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& lengths, bool batch_first) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, bool>("PackPaddedSequence"));
  return __op->call(input, lengths, batch_first);
}

Maybe<void> MultiTensorSgdUpdate(const TensorTuple& model, const TensorTuple& model_diff, double scale, float weight_decay, float learning_rate_val) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const TensorTuple&, const TensorTuple&, double, float, float>("MultiTensorSgdUpdate"));
  return __op->call(model, model_diff, scale, weight_decay, learning_rate_val);
}

Maybe<void> MultiTensorAdamUpdate(const TensorTuple& model, const TensorTuple& model_diff, const TensorTuple& m, const TensorTuple& v, float learning_rate_val, float l2, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, bool do_bias_correction, double scale, float weight_decay, float epsilon) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<void>, const TensorTuple&, const TensorTuple&, const TensorTuple&, const TensorTuple&, float, float, float, float, float, float, bool, double, float, float>("MultiTensorAdamUpdate"));
  return __op->call(model, model_diff, m, v, learning_rate_val, l2, beta1, beta2, bias_correction1_val, bias_correction2_val, do_bias_correction, scale, weight_decay, epsilon);
}

Maybe<one::Tensor> GradAccRepeat(const std::shared_ptr<one::Tensor>& input, int32_t repeat_num) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("GradAccRepeat"));
  return __op->call(input, repeat_num);
}

Maybe<one::Tensor> GradAccCollect(const std::shared_ptr<one::Tensor>& input, int32_t collect_num) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("GradAccCollect"));
  return __op->call(input, collect_num);
}

Maybe<one::Tensor> GradAccPack(const std::shared_ptr<one::Tensor>& input, int32_t pack_num) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("GradAccPack"));
  return __op->call(input, pack_num);
}

Maybe<one::Tensor> GradAccUnpack(const std::shared_ptr<one::Tensor>& input, int32_t unpack_num) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t>("GradAccUnpack"));
  return __op->call(input, unpack_num);
}

Maybe<one::Tensor> Trunc(const std::shared_ptr<one::Tensor>& input) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&>("Trunc"));
  return __op->call(input);
}

Maybe<one::Tensor> SiluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SiluGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> MishGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("MishGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SeluGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SoftSignGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SoftSignGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> GeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("GeluGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> HardSigmoidGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("HardSigmoidGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> HardSwishGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("HardSwishGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SoftplusGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double beta, double threshold) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double, double>("SoftplusGradGrad"));
  return __op->call(x, dydx, beta, threshold);
}

Maybe<one::Tensor> EluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("EluGradGrad"));
  return __op->call(x, dydx, alpha);
}

Maybe<one::Tensor> CeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double alpha) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, double>("CeluGradGrad"));
  return __op->call(x, dydx, alpha);
}

Maybe<one::TensorTuple> BatchNormStats(const std::shared_ptr<one::Tensor>& input, int32_t axis, float eps) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, int32_t, float>("BatchNormStats"));
  return __op->call(input, axis, eps);
}

Maybe<one::TensorTuple> BatchNormGatherStatsWithCounts(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const Optional<one::Tensor>& running_mean, const Optional<one::Tensor>& running_var, float momentum, float eps, const std::shared_ptr<one::Tensor>& counts) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<one::Tensor>&, float, float, const std::shared_ptr<one::Tensor>&>("BatchNormGatherStatsWithCounts"));
  return __op->call(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

Maybe<one::Tensor> BatchNormElemt(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& bias, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis, float eps) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, float>("BatchNormElemt"));
  return __op->call(input, weight, bias, mean, invstd, axis, eps);
}

Maybe<one::TensorTuple> BatchNormBackwardReduce(const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("BatchNormBackwardReduce"));
  return __op->call(grad_out, input, mean, invstd, axis);
}

Maybe<one::Tensor> BatchNormBackwardElemt(const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& sum_dy, const std::shared_ptr<one::Tensor>& sum_dy_xmu, const std::shared_ptr<one::Tensor>& count, int32_t axis) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("BatchNormBackwardElemt"));
  return __op->call(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count, axis);
}

Maybe<one::TensorTuple> AdaptiveMaxPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveMaxPool1D"));
  return __op->call(input, output_size);
}

Maybe<one::TensorTuple> AdaptiveMaxPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveMaxPool2D"));
  return __op->call(input, output_size);
}

Maybe<one::TensorTuple> AdaptiveMaxPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::vector<int64_t>&>("AdaptiveMaxPool3D"));
  return __op->call(input, output_size);
}

Maybe<one::Tensor> AdaptiveMaxPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& dy, int32_t ndims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("AdaptiveMaxPoolNdGrad"));
  return __op->call(x, index, dy, ndims);
}

Maybe<one::Tensor> TanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TanGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SinhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SinhGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> CoshGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("CoshGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> TanhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("TanhGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AcosGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AcosGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AsinGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AsinGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AtanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AtanGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AsinhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AsinhGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AcoshGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AcoshGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> AtanhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("AtanhGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> ErfGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ErfGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> ErfcGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ErfcGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> ExpGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ExpGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Expm1GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Expm1GradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> LogGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LogGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> LogSigmoidGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("LogSigmoidGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Log2GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log2GradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Log10GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log10GradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> Log1pGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("Log1pGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> ReciprocalGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReciprocalGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> ReciprocalNoNanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("ReciprocalNoNanGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> RsqrtGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("RsqrtGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SqrtGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SqrtGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SquareGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SquareGradGrad"));
  return __op->call(x, dydx);
}

Maybe<one::Tensor> SigmoidGradGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dydx) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&>("SigmoidGradGrad"));
  return __op->call(y, dydx);
}

Maybe<one::Tensor> MaxPoolNdGradGrad(const std::shared_ptr<one::Tensor>& dydx, const std::shared_ptr<one::Tensor>& indices, int32_t ndims) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t>("MaxPoolNdGradGrad"));
  return __op->call(dydx, indices, ndims);
}

Maybe<one::Tensor> Exponential(const std::shared_ptr<one::Tensor>& x, float lambd, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, float, const Optional<one::Generator>&>("Exponential"));
  return __op->call(x, lambd, generator);
}

Maybe<one::Tensor> Multinomial(const std::shared_ptr<one::Tensor>& x, int32_t num_samples, bool replacement, const Optional<one::Generator>& generator) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, int32_t, bool, const Optional<one::Generator>&>("Multinomial"));
  return __op->call(x, num_samples, replacement, generator);
}

Maybe<one::Tensor> DeformConv2d(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const std::shared_ptr<one::Tensor>& mask, const Optional<one::Tensor>& bias, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool>("DeformConv2d"));
  return __op->call(input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask);
}

Maybe<one::TensorTuple> DeformConv2dInputGrad(const std::shared_ptr<one::Tensor>& output_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const Optional<one::Tensor>& mask, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::TensorTuple>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool>("DeformConv2dInputGrad"));
  return __op->call(output_grad, input, weight, offset, mask, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask);
}

Maybe<one::Tensor> DeformConv2dParamGrad(const std::shared_ptr<one::Tensor>& output_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const std::shared_ptr<one::Tensor>& mask, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, const std::shared_ptr<one::Tensor>&, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool>("DeformConv2dParamGrad"));
  return __op->call(output_grad, input, weight, offset, mask, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask);
}

Maybe<one::Tensor> BinCount(const std::shared_ptr<one::Tensor>& input, const Optional<one::Tensor>& weights, const Optional<int64_t>& minlength) {
  static thread_local const auto& __op = CHECK_JUST(FunctionLibrary::Global()->find<Maybe<one::Tensor>, const std::shared_ptr<one::Tensor>&, const Optional<one::Tensor>&, const Optional<int64_t>&>("BinCount"));
  return __op->call(input, weights, minlength);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
