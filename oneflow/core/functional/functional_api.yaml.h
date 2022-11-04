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

#ifndef ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_
#define ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_

#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/tensor_index.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> Add(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace);

Maybe<one::Tensor> ScalarAdd(const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace);

Maybe<one::Tensor> ScalarAdd(const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha);

Maybe<one::Tensor> Add(const TensorTuple& inputs, bool inplace);

Maybe<one::Tensor> Amin(const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim);

Maybe<one::Tensor> Sub(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace);

Maybe<one::Tensor> ScalarSub(const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace);

Maybe<one::Tensor> ScalarSub(const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha);

Maybe<one::Tensor> Mul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarMul(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);

Maybe<one::Tensor> ScalarMul(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> InplaceMul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> InplaceScalarMul(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> Addcmul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);

Maybe<one::Tensor> InplaceAddcmul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);

Maybe<one::Tensor> AddCDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);

Maybe<one::Tensor> InplaceAddCDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);

Maybe<one::Tensor> Div(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarDiv(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> InplaceDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> InplaceScalarDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> DivGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& z, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> BroadcastEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastNotEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalNotEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalNotEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastGreater(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalGreater(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalGreater(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastGreaterEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalGreaterEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalGreaterEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastLogicalAnd(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalAnd(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalAnd(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastLogicalOr(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalOr(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalOr(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> LogicalNot(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> BroadcastLogicalXor(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalXor(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalXor(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastLess(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalLess(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalLess(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> BroadcastLessEqual(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarLogicalLessEqual(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> ScalarLogicalLessEqual(const Scalar& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> Pow(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& exponent);

Maybe<one::Tensor> ScalarPow(const std::shared_ptr<one::Tensor>& input, const Scalar& exponent, bool inplace);

Maybe<one::Tensor> ScalarPow(const std::shared_ptr<one::Tensor>& input, const Scalar& exponent);

Maybe<one::Tensor> ScalarReversePow(const Scalar& exponent, const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> PowXGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz);

Maybe<one::Tensor> PowYGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz);

Maybe<one::Tensor> SearchSorted(const std::shared_ptr<one::Tensor>& sorted_sequence, const std::shared_ptr<one::Tensor>& values, bool out_int32, bool right);

Maybe<one::Tensor> SearchSortedScalar(const std::shared_ptr<one::Tensor>& sorted_sequence, const Scalar& values, bool out_int32, bool right);

Maybe<one::Tensor> ScalarPowGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& dy, const Scalar& exponent);

Maybe<one::Tensor> ScalarReversePowGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& dy, const Scalar& exponent);

Maybe<one::Tensor> BroadcastPow(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> BroadcastPowXGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz);

Maybe<one::Tensor> BroadcastPowYGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dz);

Maybe<one::Tensor> FloorDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarFloorDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);

Maybe<one::Tensor> ScalarFloorDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> FloorDivXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> FloorDivYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> TruncDiv(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarTruncDiv(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);

Maybe<one::Tensor> TruncDivXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> TruncDivYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> XdivyXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> XdivyYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> XlogyXGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> XlogyYGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> Max(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> Max(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);

Maybe<one::Tensor> Max(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> Min(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> Min(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);

Maybe<one::Tensor> Min(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> Median(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> MedianWithIndices(const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);

Maybe<one::Tensor> ReduceMax(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim);

Maybe<one::Tensor> ReduceMin(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim);

Maybe<one::Tensor> ReduceSum(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);

Maybe<one::Tensor> ReduceSumWhole(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReduceNanSum(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ReduceNanSumWhole(const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ReduceMean(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);

Maybe<one::Tensor> ReduceMeanWhole(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReduceAll(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);

Maybe<one::Tensor> ReduceAllWhole(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReduceAny(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);

Maybe<one::Tensor> ReduceAnyWhole(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReduceProd(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ReduceProdWhole(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype);

Maybe<one::TensorTuple> ReduceMinDeviceStage(const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis);

Maybe<one::Tensor> ReduceMinDeviceStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& count, const std::vector<int32_t>& axis);

Maybe<one::TensorTuple> ReduceMaxDeviceStage(const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis);

Maybe<one::Tensor> ReduceMaxDeviceStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& count, const std::vector<int32_t>& axis);

Maybe<one::TensorTuple> ReduceMinGlobalStage(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);

Maybe<one::Tensor> ReduceMinGlobalStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);

Maybe<one::TensorTuple> ReduceMaxGlobalStage(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);

Maybe<one::Tensor> ReduceMaxGlobalStageGrad(const std::shared_ptr<one::Tensor>& out_diff, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);

Maybe<one::Tensor> Transpose(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& perm);

Maybe<one::Tensor> Transpose2dim(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);

Maybe<one::Tensor> AsStrided(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size, const std::vector<int32_t>& stride, int32_t storage_offset);

Maybe<one::Tensor> AsStridedGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size, const std::vector<int32_t>& stride, int32_t storage_offset);

Maybe<one::Tensor> Select(const std::shared_ptr<one::Tensor>& input, int32_t dim, int32_t index);

Maybe<one::Tensor> Swapaxes(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);

Maybe<one::Tensor> Swapdims(const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);

Maybe<one::Tensor> Amax(const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim);

Maybe<one::Tensor> Permute(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dims);

Maybe<one::Tensor> TransposeAllDimProperty(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> TransposeAllDimFunction(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> NotEqualZero(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> NotEqualZeroGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Reciprocal(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReciprocalGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> ReciprocalNoNan(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ReciprocalNoNanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> ImageFlip(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& flip_code);

Maybe<one::Tensor> Sin(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SinGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> SinGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Sin_(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Cos(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> CosGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> CosGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Cosh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> CoshGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> BroadcastFMod(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> ScalarFMod(const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);

Maybe<one::Tensor> ScalarFMod(const std::shared_ptr<one::Tensor>& input, const Scalar& other);

Maybe<one::Tensor> Log(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LogGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Log2(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Log2Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Log10(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Log10Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Sqrt(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SqrtGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Rsqrt(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> RsqrtGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Square(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SquareGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> SqrtSquareSum(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> StandardDeviation(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim);

Maybe<one::Tensor> Variance(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim);

Maybe<one::Tensor> RMSLayerNormalization(const std::shared_ptr<one::Tensor>& hidden_states, const std::shared_ptr<one::Tensor>& weight, float variance_epsilon);

Maybe<one::Tensor> Relu(const std::shared_ptr<one::Tensor>& x, bool inplace);

Maybe<one::Tensor> ReluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> HannWindow(int64_t window_length, bool periodic, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool requires_grad);

Maybe<one::Tensor> GlobalHannWindow(int64_t window_length, bool periodic, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, bool requires_grad);

Maybe<one::Tensor> HardTanh(const std::shared_ptr<one::Tensor>& x, double min_val, double max_val);

Maybe<one::Tensor> HardTanhGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double min_val, double max_val);

Maybe<one::Tensor> Tan(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> TanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Tanh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> TanhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Threshold(const std::shared_ptr<one::Tensor>& x, double threshold, double value);

Maybe<one::Tensor> ThresholdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double threshold);

Maybe<one::Tensor> Elu(const std::shared_ptr<one::Tensor>& x, double alpha);

Maybe<one::Tensor> EluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double alpha);

Maybe<one::Tensor> Celu(const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace);

Maybe<one::Tensor> CeluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double alpha);

Maybe<one::Tensor> Gelu(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> GeluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> GeluWithApproximate(const std::shared_ptr<one::Tensor>& x, const std::string& approximate);

Maybe<one::Tensor> Glu(const std::shared_ptr<one::Tensor>& input, int64_t dim);

Maybe<one::Tensor> Sigmoid(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SigmoidGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> HardSigmoid(const std::shared_ptr<one::Tensor>& input, bool inplace);

Maybe<one::Tensor> HardSigmoidGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> HardShrink(const std::shared_ptr<one::Tensor>& x, double lambd, bool inplace);

Maybe<one::Tensor> HardShrinkGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double lambd);

Maybe<one::Tensor> Softmax(const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim);

Maybe<one::Tensor> SoftmaxGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> LogSoftmax(const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim);

Maybe<one::Tensor> LogSoftmaxGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> HardSwish(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> HardSwishGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LeakyRelu(const std::shared_ptr<one::Tensor>& x, float alpha, bool inplace);

Maybe<one::Tensor> LeakyReluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, float alpha);

Maybe<one::Tensor> Normal(float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> Normal2(float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalNormal(float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalNormal2(float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> Normalization(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training);

Maybe<one::TensorTuple> NormalizationGrad(const std::shared_ptr<one::Tensor>& grad, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const std::shared_ptr<one::Tensor>& gamma, float epsilon, int32_t axis);

Maybe<one::Tensor> NormalizationAddRelu(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& addend, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training);

Maybe<one::TensorTuple> NormalizationAddReluGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& moving_mean, const std::shared_ptr<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, const std::shared_ptr<one::Tensor>& reserve_space, const std::shared_ptr<one::Tensor>& y, int32_t axis, float epsilon, bool has_addend);

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad);

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const std::string& device, bool requires_grad);

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);

Maybe<one::Tensor> Eye(const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const Symbol<SbpParallel>& sbp);

Maybe<one::Tensor> EyeInplace(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Erfinv(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ErfinvInplace(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Arange(const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device);

Maybe<one::Tensor> Arange(const Scalar& end, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device);

Maybe<one::Tensor> GlobalArange(const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);

Maybe<one::Tensor> GlobalArange(const Scalar& end, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);

Maybe<one::Tensor> Flatten(const std::shared_ptr<one::Tensor>& x, int32_t start_dim, int32_t end_dim);

Maybe<one::Tensor> ArgMax(const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ArgMin(const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::TensorTuple> ArgWhere(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype);

Maybe<one::TensorTuple> NonZero(const std::shared_ptr<one::Tensor>& x, bool as_tuple);

Maybe<one::Tensor> BroadcastLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& broadcast_axes);

Maybe<one::Tensor> Cast(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype, bool pin_memory);

Maybe<one::Tensor> Constant(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device);

Maybe<one::Tensor> GlobalConstant(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);

Maybe<one::Tensor> Empty(const Shape& shape, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool pin_memory);

Maybe<one::Tensor> GlobalEmpty(const Shape& shape, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);

Maybe<one::Tensor> ZerosLike(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> OnesLike(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Bernoulli(const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace);

Maybe<one::Tensor> BernoulliProb(const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace);

Maybe<one::Tensor> BernoulliInplace(const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator);

Maybe<one::Tensor> BernoulliProbInplace(const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator);

Maybe<one::Tensor> Concat(const TensorTuple& inputs, int64_t dim);

Maybe<one::Tensor> BiasAdd(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& bias, int32_t axis);

Maybe<one::Tensor> Conv1d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);

Maybe<one::Tensor> Conv2d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);

Maybe<one::Tensor> Conv3d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);

Maybe<one::Tensor> FakeQuantization(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme);

Maybe<one::Tensor> Quantization(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme);

Maybe<one::TensorTuple> MinMaxObserver(const std::shared_ptr<one::Tensor>& in, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, bool per_layer_quantization);

Maybe<one::TensorTuple> MovingAverageMinMaxObserver(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& current_train_step, const std::shared_ptr<one::Tensor>& moving_max, const std::shared_ptr<one::Tensor>& moving_min, bool training, int64_t stop_update_after_iters, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, float momentum);

Maybe<one::Tensor> ConvDataGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x, int32_t num_spatial_dims, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& strides, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& dilation_rate, int32_t groups, const std::string& data_format);

Maybe<one::Tensor> ConvFilterGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, int32_t num_spatial_dims, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& strides, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& dilation_rate, int32_t groups, const std::string& data_format);

Maybe<one::Tensor> ConvBiasGrad(const std::shared_ptr<one::Tensor>& dy, int32_t num_spatial_dims, const std::string& data_format);

Maybe<one::Tensor> Deconv1d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);

Maybe<one::Tensor> Deconv2d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);

Maybe<one::Tensor> Deconv3d(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);

Maybe<one::Tensor> Expand(const std::shared_ptr<one::Tensor>& x, const Shape& shape);

Maybe<one::Tensor> Repeat(const std::shared_ptr<one::Tensor>& input, const Shape& repeat_shape);

Maybe<one::Tensor> RepeatInterLeaveIndex(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& cumsum, int32_t dim);

Maybe<one::Tensor> RepeatInterLeaveInt(const std::shared_ptr<one::Tensor>& input, int32_t repeats, const Optional<int32_t>& dim);

Maybe<one::Tensor> RepeatInterLeaveTensor(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& repeats, int32_t dim, const Optional<int32_t>& output_size);

Maybe<one::Tensor> Tile(const std::shared_ptr<one::Tensor>& input, const Shape& dims);

Maybe<one::Tensor> Roll(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& shifts, const Optional<std::vector<int32_t>>& dims);

Maybe<one::Tensor> ExpandDims(const std::shared_ptr<one::Tensor>& input, int32_t dim);

Maybe<one::Tensor> Unsqueeze(const std::shared_ptr<one::Tensor>& input, int32_t dim);

Maybe<one::Tensor> UnsqueezeMultiple(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dim, int32_t dims);

Maybe<one::Tensor> Squeeze(const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim);

Maybe<one::Tensor> Exp(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ExpGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Gather(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& indices, int64_t axis);

Maybe<one::Tensor> DimGather(const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index, bool sparse_grad);

Maybe<one::Tensor> EmbeddingReNorm(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices, double max_norm, double norm_type);

Maybe<one::Tensor> Embedding(const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& indices, const Optional<int64_t>& padding_idx, bool scale_grad_by_freq);

Maybe<one::Tensor> EmbeddingGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& indices, int64_t padding_idx, bool scale_grad_by_freq);

Maybe<one::Tensor> ArgSort(const std::shared_ptr<one::Tensor>& in, const std::string& direction);

Maybe<one::Tensor> GatherNd(const std::shared_ptr<one::Tensor>& params, const std::shared_ptr<one::Tensor>& indices);

Maybe<one::Tensor> ScatterNd(const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, const Shape& shape);

Maybe<one::Tensor> TensorScatterNdUpdate(const std::shared_ptr<one::Tensor>& tensor, const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, bool inplace);

Maybe<one::Tensor> ScatterNdLike(const std::shared_ptr<one::Tensor>& like, const std::shared_ptr<one::Tensor>& updates, const std::shared_ptr<one::Tensor>& indices);

Maybe<one::Tensor> MatMul(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, bool transpose_a, bool transpose_b, double alpha);

Maybe<one::Tensor> MatMulNoBroadCast(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mat2);

Maybe<one::Tensor> FusedMLP(const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation);

Maybe<one::TensorTuple> FusedMLPGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& cublas_aux, const TensorTuple& hidden, const std::vector<float>& alpha_list);

Maybe<one::TensorTuple> CublasBiasAddReluMatmulGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& aux, double alpha);

Maybe<one::TensorTuple> CublasMatmulBiasAddGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> FusedMatmulBiasAddReluDropout(const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation, const std::vector<float>& dropout_rate_list, const Optional<one::Generator>& generator);

Maybe<one::Tensor> FusedReluDropoutGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale);

Maybe<one::Tensor> BroadcastMatmulGradB(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, double alpha);

Maybe<one::Tensor> BatchMatMul(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, bool transpose_a, bool transpose_b, double alpha);

Maybe<one::Tensor> MatrixVectorProduct(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& vec);

Maybe<one::Tensor> MatrixVectorProductGradA(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& b);

Maybe<one::Tensor> MatrixVectorProductGradB(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& a);

Maybe<one::Tensor> VectorMatrixProduct(const std::shared_ptr<one::Tensor>& vec, const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> VectorMatrixProductGradA(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& b);

Maybe<one::Tensor> VectorMatrixProductGradB(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& a);

Maybe<one::Tensor> TensorDot(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, const std::vector<int32_t>& dims_a, const std::vector<int32_t>& dims_b);

Maybe<one::Tensor> TensorDotIntDims(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t dims);

Maybe<one::Tensor> L1Loss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction);

Maybe<one::Tensor> MseLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction);

Maybe<one::Tensor> KLDivLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target, const std::string& reduction);

Maybe<one::Tensor> KLDivLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target);

Maybe<one::Tensor> KLDivLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target);

Maybe<one::Tensor> NLLLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction);

Maybe<one::Tensor> NLLGrad(const std::shared_ptr<one::Tensor>& out_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index);

Maybe<one::Tensor> BinaryCrossEntropyLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const std::string& reduction);

Maybe<one::Tensor> BinaryCrossEntropyLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight);

Maybe<one::Tensor> BinaryCrossEntropyLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight);

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLoss(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight, const std::string& reduction);

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight);

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight);

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsReduceMeanLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target);

Maybe<one::Tensor> BinaryCrossEntropyWithLogitsReduceMeanLossTargetGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target);

Maybe<one::Tensor> SparseCrossEntropy(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth);

Maybe<one::Tensor> SparseCrossEntropyGrad(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& dy, int64_t depth);

Maybe<one::Tensor> SparseCrossEntropyMs(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth);

Maybe<one::Tensor> CrossEntropy(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction, double label_smoothing);

Maybe<one::Tensor> CrossEntropyLabelSmoothing(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction, double label_smoothing);

Maybe<one::Tensor> CrossEntropyProb(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const std::string& reduction, double label_smoothing);

Maybe<one::Tensor> SparseCrossEntropyMsGrad(const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& dy, int64_t depth);

Maybe<one::Tensor> SparseSoftmaxCrossEntropy(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label);

Maybe<one::Tensor> SparseSoftmaxCrossEntropyGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& prob, const std::shared_ptr<one::Tensor>& label, int64_t depth);

Maybe<one::Tensor> SparseSoftmaxCrossEntropyMsGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& prob, const std::shared_ptr<one::Tensor>& label, int64_t depth);

Maybe<one::Tensor> SoftmaxCrossEntropy(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label);

Maybe<one::Tensor> SoftmaxCrossEntropyGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& prob);

Maybe<one::Tensor> SmoothL1Loss(const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label, float beta, const std::string& reduction);

Maybe<one::Tensor> SmoothL1LossGrad(const std::shared_ptr<one::Tensor>& loss_grad, const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, float beta);

Maybe<one::Tensor> CombinedMarginLoss(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& label, float m1, float m2, float m3);

Maybe<one::Tensor> CombinedMarginLossGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& theta, float m1, float m2, float m3, int64_t depth);

Maybe<one::Tensor> TripletMarginLoss(const std::shared_ptr<one::Tensor>& anchor, const std::shared_ptr<one::Tensor>& positive, const std::shared_ptr<one::Tensor>& negative, float margin, float p, float eps, bool swap, const std::string& reduction);

Maybe<one::Tensor> MarginRankingLoss(const std::shared_ptr<one::Tensor>& input_1, const std::shared_ptr<one::Tensor>& input_2, const std::shared_ptr<one::Tensor>& target, float margin, const std::string& reduction);

Maybe<one::Tensor> CtcLoss(const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& input_lengths, const std::shared_ptr<one::Tensor>& target_lengths, int64_t max_target_length, int64_t blank, bool zero_infinity, const std::string& reduction);

Maybe<one::Tensor> AffineGrid(const std::shared_ptr<one::Tensor>& theta, const Shape& size, bool align_corners);

Maybe<one::Tensor> AffineGridGrad(const std::shared_ptr<one::Tensor>& dgrid, const Shape& size, bool align_corners);

Maybe<one::Tensor> GridSample(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& grid, const std::string& interpolation_mode, const std::string& padding_mode, bool align_corners);

Maybe<one::TensorTuple> GridSampleGrad(const std::shared_ptr<one::Tensor>& doutput, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& grid, const std::string& interpolation_mode, const std::string& padding_mode, bool align_corners);

Maybe<one::Tensor> Where(const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> WhereScalarX(const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> WhereScalarY(const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const Scalar& y);

Maybe<one::Tensor> WhereScalarXY(const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const Scalar& y);

Maybe<one::Tensor> MaskedFill(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value);

Maybe<one::Tensor> MaskedFillInplace(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value);

Maybe<one::Tensor> MovedimInt(const std::shared_ptr<one::Tensor>& input, int32_t source, int32_t destination);

Maybe<one::Tensor> MovedimVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& source, const std::vector<int32_t>& destination);

Maybe<one::TensorTuple> TensorSplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections, int32_t dim);

Maybe<one::TensorTuple> TensorSplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections, int32_t dim);

Maybe<one::TensorTuple> HsplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections);

Maybe<one::TensorTuple> HsplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections);

Maybe<one::TensorTuple> VsplitInt(const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections);

Maybe<one::TensorTuple> VsplitVec(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections);

Maybe<one::Tensor> Negative(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LayerNormAffine(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon);

Maybe<one::Tensor> LayerNorm(const std::shared_ptr<one::Tensor>& x, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon);

Maybe<one::Tensor> LayerNormGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, int64_t begin_norm_axis, double epsilon);

Maybe<one::Tensor> LayerNormAffineGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const std::shared_ptr<one::Tensor>& gamma, int64_t begin_norm_axis, double epsilon);

Maybe<one::TensorTuple> LayerNormParamGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, int64_t begin_params_axis);

Maybe<one::Tensor> GroupNorm(const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, bool affine, int32_t num_groups, double epsilon);

Maybe<one::Tensor> GroupNormGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance, const Optional<one::Tensor>& gamma, int32_t num_groups, double epsilon);

Maybe<one::TensorTuple> GroupNormParamGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& inv_variance);

Maybe<one::Tensor> TFAvgPool2D(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, const std::string& padding, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& padding_after, const std::string& data_format, bool ceil_mode);

Maybe<one::Tensor> CtcLossGrad(const std::shared_ptr<one::Tensor>& loss_grad, const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& input_lengths, const std::shared_ptr<one::Tensor>& target_lengths, const std::shared_ptr<one::Tensor>& loss, const std::shared_ptr<one::Tensor>& alpha, int64_t blank, bool zero_infinity, int64_t max_target_length);

Maybe<one::Tensor> AdaptiveAvgPool1D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);

Maybe<one::Tensor> AdaptiveAvgPool2D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);

Maybe<one::Tensor> AdaptiveAvgPool3D(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);

Maybe<one::Tensor> AdaptivePoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, const std::string& mode, int32_t ndims);

Maybe<one::Tensor> TFPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, const std::string& mode, int32_t ndims, const std::string& data_format, const std::string& padding, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& padding_after, const std::vector<int32_t>& pool_size, const std::vector<int32_t>& strides, bool ceil_mode);

Maybe<one::TensorTuple> MaxPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);

Maybe<one::TensorTuple> MaxPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);

Maybe<one::TensorTuple> MaxPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);

Maybe<one::Tensor> MaxPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& indice, const std::shared_ptr<one::Tensor>& dy, int32_t ndims, const std::string& data_format, const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode);

Maybe<one::Tensor> PRelu(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& alpha);

Maybe<one::TensorTuple> PReluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& alpha);

Maybe<one::Tensor> Reshape(const std::shared_ptr<one::Tensor>& x, const Shape& shape);

Maybe<one::Tensor> View(const std::shared_ptr<one::Tensor>& x, const Shape& shape);

Maybe<one::Tensor> ToContiguous(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> InplaceToContiguous(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> SliceView1dContiguous(const std::shared_ptr<one::Tensor>& x, int64_t start, int64_t end);

Maybe<one::Tensor> Narrow(const std::shared_ptr<one::Tensor>& input, int64_t dim, int64_t start, int64_t length);

Maybe<one::Tensor> NarrowGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& like, int64_t dim, int64_t start, int64_t length);

Maybe<one::Tensor> Slice(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, const Optional<bool>& enable_view_slice);

Maybe<one::Tensor> SliceUpdate(const std::shared_ptr<one::Tensor>& ref, const std::shared_ptr<one::Tensor>& value, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, bool inplace);

Maybe<one::Tensor> SliceGrad(const std::shared_ptr<one::Tensor>& dy, const Shape& like_shape, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step);

Maybe<one::Tensor> Copy(const std::shared_ptr<one::Tensor>& x, const std::string& device_type, int64_t device_id, bool pin_memory);

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device, const Optional<Symbol<DType>>& dtype, bool copy);

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool copy);

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype, bool copy);

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& other, bool copy);

Maybe<one::Tensor> To(const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device);

Maybe<one::Tensor> Flip(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dims);

Maybe<one::Tensor> Upsample(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const std::string& interpolation, const std::string& data_format);

Maybe<one::Tensor> UpsampleGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const std::string& data_format, const std::string& interpolation);

Maybe<one::Tensor> UpsampleLinear1D(const std::shared_ptr<one::Tensor>& x, double scale_factor, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleLinear1DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double scale_factor, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest1D(const std::shared_ptr<one::Tensor>& x, double scale_factor, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest1DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double scale_factor, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleBilinear2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleBilinear2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleBicubic2D(const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleBicubic2DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest3D(const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleNearest3DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleTrilinear3D(const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> UpsampleTrilinear3DGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);

Maybe<one::Tensor> Abs(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AbsGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Acos(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AcosGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Acosh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AcoshGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Asin(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AsinGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Asinh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AsinhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Atan(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AtanGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Atan2(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> Atan2XGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> Atan2YGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> Atanh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> AtanhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Ceil(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> CeilGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Erf(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ErfGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Erfc(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> ErfcGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Expm1(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Expm1Grad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Floor(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Floor_(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> FloorGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Lgamma(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LgammaGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Log1p(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Log1pGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> LogSigmoid(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LogSigmoidGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Rint(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> RintGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Round(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> RoundGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Sign(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SignGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Sinh(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SinhGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);

Maybe<one::Tensor> Softplus(const std::shared_ptr<one::Tensor>& x, double beta, double threshold);

Maybe<one::Tensor> SoftplusGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, double beta, double threshold);

Maybe<one::Tensor> SoftShrink(const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace);

Maybe<one::Tensor> SoftShrinkGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, double alpha);

Maybe<one::Tensor> OneHot(const std::shared_ptr<one::Tensor>& input, int64_t num_classes, const Scalar& on_value, const Scalar& off_value);

Maybe<one::Tensor> UnsortedSegmentSumLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& segment_ids, const std::shared_ptr<one::Tensor>& like, int64_t axis);

Maybe<one::Tensor> UnsortedSegmentSum(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& segment_ids, int64_t axis, int64_t num_segments);

Maybe<one::Tensor> Tril(const std::shared_ptr<one::Tensor>& x, int64_t diagonal);

Maybe<one::Tensor> Triu(const std::shared_ptr<one::Tensor>& x, int64_t diagonal);

Maybe<one::Tensor> InplaceTriu(const std::shared_ptr<one::Tensor>& x, int64_t diagonal);

Maybe<one::Tensor> Clamp(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);

Maybe<one::Tensor> ClampInplace(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);

Maybe<one::Tensor> ClampMin(const std::shared_ptr<one::Tensor>& input, const Scalar& min);

Maybe<one::Tensor> ClampMinInplace(const std::shared_ptr<one::Tensor>& input, const Scalar& min);

Maybe<one::Tensor> ClampMax(const std::shared_ptr<one::Tensor>& input, const Scalar& max);

Maybe<one::Tensor> ClampMaxInplace(const std::shared_ptr<one::Tensor>& input, const Scalar& min);

Maybe<one::Tensor> Clip(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);

Maybe<one::Tensor> ClipInplace(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);

Maybe<one::Tensor> ClampGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min, const Optional<Scalar>& max);

Maybe<one::Tensor> VectorNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> VectorNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> MatrixNorm(const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> MatrixNorm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> Norm(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype, bool for_norm);

Maybe<one::Tensor> Norm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ScalarNorm(const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> ScalarNorm(const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> Inv(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> LinalgCross(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Optional<int64_t>& dim);

Maybe<one::Tensor> Dropout(const std::shared_ptr<one::Tensor>& input, float p, bool training, bool inplace, const Optional<one::Generator>& generator, const Optional<one::Tensor>& addend);

Maybe<one::Tensor> DropoutGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale);

Maybe<one::Tensor> Dropout1d(const std::shared_ptr<one::Tensor>& input, float p, bool training);

Maybe<one::Tensor> Dropout2d(const std::shared_ptr<one::Tensor>& input, float p, bool training);

Maybe<one::Tensor> Dropout3d(const std::shared_ptr<one::Tensor>& input, float p, bool training);

Maybe<one::Tensor> ConstantPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad, const Scalar& value);

Maybe<one::Tensor> ReflectionPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad);

Maybe<one::Tensor> ReplicationPad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad);

Maybe<one::Tensor> Pad(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad, const std::string& mode, const Scalar& value);

Maybe<one::Tensor> PadGrad(const std::shared_ptr<one::Tensor>& dy, const std::vector<int64_t>& pad, const std::string& mode, const Scalar& value);

Maybe<one::Tensor> Silu(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SiluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Mish(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> MishGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Selu(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SeluGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SoftSign(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> SoftSignGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> Diag(const std::shared_ptr<one::Tensor>& x, int32_t diagonal);

Maybe<one::Tensor> DiagGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& in, int32_t diagonal);

Maybe<one::Tensor> Diagonal(const std::shared_ptr<one::Tensor>& x, int32_t offset, int32_t dim1, int32_t dim2);

Maybe<one::Tensor> DiagonalGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& in, int32_t offset);

Maybe<one::Tensor> TensorGetItem(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index);

Maybe<one::Tensor> DimScatter(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, const Optional<std::string>& reduce, bool inplace);

Maybe<one::Tensor> DimScatterScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, const Optional<std::string>& reduce, bool inplace);

Maybe<one::Tensor> DimScatterUpdate(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace);

Maybe<one::Tensor> DimScatterUpdateScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace);

Maybe<one::Tensor> DimScatterAdd(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace);

Maybe<one::Tensor> DimScatterAddScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace);

Maybe<one::Tensor> DimScatterMul(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace);

Maybe<one::Tensor> DimScatterMulScalar(const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace);

Maybe<one::Tensor> DimScatterAddLike(const std::shared_ptr<one::Tensor>& like, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src);

Maybe<void> TensorSetItem(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index, const std::shared_ptr<one::Tensor>& value);

Maybe<one::Tensor> AvgPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);

Maybe<one::Tensor> AvgPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);

Maybe<one::Tensor> AvgPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);

Maybe<one::Tensor> AvgPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy, int32_t ndims, const std::string& data_format, const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, bool ceil_mode, bool count_include_pad, int32_t divisor_override);

Maybe<one::Tensor> Minimum(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> Maximum(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::TensorTuple> ElementwiseMinGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::TensorTuple> ElementwiseMaxGrad(const std::shared_ptr<one::Tensor>& dz, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);

Maybe<one::Tensor> Stack(const TensorTuple& inputs, int64_t dim);

Maybe<one::TensorTuple> StackGrad(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like, int64_t axis);

Maybe<one::Tensor> AtLeast1D(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> AtLeast1D(const TensorTuple& tensors);

Maybe<one::Tensor> AtLeast2D(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> AtLeast2D(const TensorTuple& tensors);

Maybe<one::Tensor> AtLeast3D(const std::shared_ptr<one::Tensor>& input);

Maybe<one::TensorTuple> AtLeast3D(const TensorTuple& tensors);

Maybe<one::Tensor> HStack(const TensorTuple& tensors);

Maybe<one::Tensor> VStack(const TensorTuple& tensors);

Maybe<one::Tensor> DStack(const TensorTuple& tensors);

Maybe<one::Tensor> ColumnStack(const TensorTuple& tensors);

Maybe<one::Tensor> RowStack(const TensorTuple& tensors);

Maybe<one::Tensor> LocalToGlobal(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Shape& shape, const Symbol<DType>& dtype, bool sync_data, bool copy);

Maybe<one::Tensor> ToGlobal(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const std::vector<Symbol<SbpParallel>>& grad_sbp, bool check_meta, bool copy);

Maybe<one::Tensor> GlobalToLocal(const std::shared_ptr<one::Tensor>& x, bool copy);

Maybe<void> StreamTouch(const TensorTuple& x);

Maybe<one::TensorTuple> BroadcastTensors(const TensorTuple& inputs, int64_t src_rank, bool inplace);

Maybe<one::Tensor> LocalAllReduce(const std::shared_ptr<one::Tensor>& x, bool inplace);

Maybe<one::Tensor> LocalReduce(const std::shared_ptr<one::Tensor>& x, int64_t dst, bool inplace);

Maybe<one::Tensor> EagerPToB(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const Shape& shape);

Maybe<one::Tensor> EagerBToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape);

Maybe<one::Tensor> EagerSToB(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& in_sbp, const Shape& shape);

Maybe<one::Tensor> EagerNaiveSToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& in_sbp, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape);

Maybe<one::Tensor> EagerPToS(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape);

Maybe<one::Tensor> EagerSToP(const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& in_placement, const Symbol<ParallelDesc>& out_placement, const std::vector<Symbol<SbpParallel>>& out_sbp, const Shape& shape);

Maybe<one::Tensor> GlobalAllReduce(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> GlobalReduceScatter(const std::shared_ptr<one::Tensor>& x, const std::string& op_type);

Maybe<one::Tensor> GlobalAllGather(const std::shared_ptr<one::Tensor>& x);

Maybe<one::Tensor> GlobalS2S(const std::shared_ptr<one::Tensor>& x, const std::vector<Symbol<SbpParallel>>& out_sbp);

Maybe<one::TensorTuple> SelectTopN(const TensorTuple& inputs, int32_t n);

Maybe<one::Tensor> CastLike(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& like);

Maybe<one::Tensor> Identity(const std::shared_ptr<one::Tensor>& in);

Maybe<one::Tensor> AmpWhiteIdentity(const std::shared_ptr<one::Tensor>& in);

Maybe<one::Tensor> AmpBlackIdentity(const std::shared_ptr<one::Tensor>& in);

Maybe<one::Tensor> ReshapeLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like);

Maybe<one::Tensor> ReduceSumLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& axis);

Maybe<one::Tensor> BroadcastReduceSumLike(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like);

Maybe<one::Tensor> Rand(const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRand(const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandN(const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandN(const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandnLike(const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandnLike(const std::shared_ptr<one::Tensor>& input, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandInt(int64_t low, int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandInt(int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandInt(int64_t low, int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandInt(int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> GlobalRandIntLike(const std::shared_ptr<one::Tensor>& x, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);

Maybe<one::Tensor> RandPerm(int32_t n, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad);

Maybe<one::Tensor> GlobalRandPerm(int32_t n, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, bool requires_grad);

Maybe<one::Tensor> UnfoldTensor(const std::shared_ptr<one::Tensor>& x, int32_t dimension, int32_t size, int32_t step);

Maybe<one::Tensor> UnfoldTensorGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x, int32_t dimension, int32_t size, int32_t step);

Maybe<one::Tensor> Unfold(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format);

Maybe<one::Tensor> Fold(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& output_size, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format);

Maybe<one::TensorTuple> Split(const std::shared_ptr<one::Tensor>& x, int64_t split_size_or_sections, int64_t dim);

Maybe<one::TensorTuple> SplitWithSize(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& split_size_or_sections, int64_t dim);

Maybe<one::TensorTuple> Unbind(const std::shared_ptr<one::Tensor>& x, int64_t dim);

Maybe<one::TensorTuple> Chunk(const std::shared_ptr<one::Tensor>& x, int64_t chunks, int64_t dim);

Maybe<one::TensorTuple> SplitLike(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like, int64_t axis);

Maybe<one::Tensor> PairwiseDistance(const std::shared_ptr<one::Tensor>& x1, const std::shared_ptr<one::Tensor>& x2, float p, double eps, bool keepdim);

Maybe<one::Tensor> CosineSimilarity(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, int32_t dim, double eps);

Maybe<one::Tensor> Normalize(const std::shared_ptr<one::Tensor>& input, float p, int32_t dim, float eps, bool use_l2_norm_kernel);

Maybe<one::Tensor> L2Normalize(const std::shared_ptr<one::Tensor>& input, int32_t axis, float epsilon);

Maybe<one::Tensor> L2NormalizeGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& square_x_sum, int32_t axis, float epsilon);

Maybe<one::TensorTuple> FusedSelfAttention(const std::shared_ptr<one::Tensor>& hidden_states, int64_t head_size, float alpha);

Maybe<one::Tensor> FusedSelfAttentionGrad(const std::shared_ptr<one::Tensor>& query_mul_key_grad, const std::shared_ptr<one::Tensor>& value_grad, const std::shared_ptr<one::Tensor>& hidden_states, float alpha);

Maybe<one::Tensor> FusedScaleTril(const std::shared_ptr<one::Tensor>& x, int64_t diagonal, const Scalar& fill_value, const Scalar& scale);

Maybe<one::Tensor> FusedBiasAddGelu(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t axis);

Maybe<one::Tensor> FusedBiasAddGeluGrad(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, const std::shared_ptr<one::Tensor>& dy, int32_t axis);

Maybe<one::Tensor> FusedBiasAddDropout(const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, float p, int32_t axis, const Optional<one::Generator>& generator);

Maybe<one::Tensor> FusedScaleMaskSoftmax(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale);

Maybe<one::Tensor> FusedScaleMaskSoftmaxGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, float scale);

Maybe<one::TensorTuple> FusedScaleMaskSoftmaxDropout(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale, float p, bool training, const Optional<one::Generator>& generator);

Maybe<one::Tensor> FusedScaleMaskSoftmaxDropoutGrad(const std::shared_ptr<one::Tensor>& softmax_y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, const std::shared_ptr<one::Tensor>& dropout_mask, float scale, float dropout_scale);

Maybe<one::TensorTuple> FusedScaleTrilSoftmaxMaskScale(const std::shared_ptr<one::Tensor>& a, float p, int64_t diagonal, float tril_scale_value, float tril_fill_value, const Optional<one::Generator>& generator);

Maybe<one::Tensor> FusedScaleTrilSoftmaxMaskScaleGrad(const std::shared_ptr<one::Tensor>& softmax_y, const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& mask, int64_t diagonal, float tril_scale_value, float mask_scale_value);

Maybe<one::Tensor> FusedMultiHeadAttentionInference(const std::shared_ptr<one::Tensor>& query, const std::shared_ptr<one::Tensor>& key, const std::shared_ptr<one::Tensor>& value, int64_t num_heads, bool causal, int64_t query_hidden_slice_start, int64_t query_hidden_slice_end, int64_t key_hidden_slice_start, int64_t key_hidden_slice_end, int64_t value_hidden_slice_start, int64_t value_hidden_slice_end);

Maybe<void> Send(const std::shared_ptr<one::Tensor>& input, int64_t dst, bool send_meta);

Maybe<one::Tensor> Recv(int64_t src, const Optional<Shape>& shape, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Tensor>& out);

Maybe<one::Tensor> BatchGather(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices);

Maybe<one::Tensor> UnsortedBatchSegmentSum(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& segment_ids, int64_t num_segments);

Maybe<one::TensorTuple> CtcGreedyDecoder(const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& input_lengths, bool merge_repeated);

Maybe<one::TensorTuple> DistributedPariticalFCSampleDisableBoxing(const std::shared_ptr<one::Tensor>& sampled_weight_diff, const std::shared_ptr<one::Tensor>& sampled_label);

Maybe<one::Tensor> Nms(const std::shared_ptr<one::Tensor>& x, float iou_threshold, int32_t keep_n);

Maybe<one::Tensor> RoiAlign(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& rois, float spatial_scale, int32_t pooled_h, int32_t pooled_w, int32_t sampling_ratio, bool aligned);

Maybe<one::Tensor> RoiAlignGrad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& x_like, const std::shared_ptr<one::Tensor>& rois, float spatial_scale, int32_t pooled_h, int32_t pooled_w, int32_t sampling_ratio, bool aligned);

Maybe<one::TensorTuple> Meshgrid(const TensorTuple& tensors, const std::string& indexing);

Maybe<one::Tensor> IndexSelect(const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index);

Maybe<one::Tensor> DecodeOneRec(const std::shared_ptr<one::Tensor>& input, const std::string& key, const Symbol<DType>& dtype, const Shape& shape, bool is_dynamic, const Optional<Shape>& reshape, const Optional<Shape>& batch_padding);

Maybe<one::Tensor> Dot(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);

Maybe<one::Tensor> FusedDotFeatureInteraction(const TensorTuple& features, const Optional<one::Tensor>& output_concat, bool self_interaction, int32_t output_padding, const std::string& pooling);

Maybe<one::TensorTuple> FusedDotFeatureInteractionGrad(const std::shared_ptr<one::Tensor>& dy, const TensorTuple& features, bool has_output_concat_grad, bool self_interaction, int32_t output_concat_grad_dim, const std::string& pooling);

Maybe<one::Tensor> FusedCrossFeatureInteraction(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& bias, const std::string& interaction_mode);

Maybe<one::TensorTuple> FusedCrossFeatureInteractionV1Grad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& matmul_result);

Maybe<one::TensorTuple> FusedCrossFeatureInteractionV2Grad(const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& bias, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& matmul_result);

Maybe<one::Tensor> TensorBufferToTensor(const std::shared_ptr<one::Tensor>& input, const Shape& instance_shape, const Symbol<DType>& dtype);

Maybe<one::Tensor> TensorToTensorBuffer(const std::shared_ptr<one::Tensor>& input, int32_t instance_dims);

Maybe<one::Tensor> GenTensorBuffer(const Shape& shape, const std::vector<Shape>& shape_list, const std::vector<float>& value_list, const Symbol<DType>& data_type, bool dynamic_out);

Maybe<one::Tensor> TopK(const std::shared_ptr<one::Tensor>& input, int32_t k, bool sorted);

Maybe<one::Tensor> InTopK(const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& predictions, int32_t k);

Maybe<one::Tensor> Cumsum(const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> Cumprod(const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype);

Maybe<one::Tensor> CumprodGrad(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& x, int64_t dim);

Maybe<one::TensorTuple> OneEmbeddingIdShuffle(const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, int32_t num_tables, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingEmbeddingShuffle(const std::shared_ptr<one::Tensor>& cur_rank_embeddings, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingEmbeddingGradientShuffle(const std::shared_ptr<one::Tensor>& embedding_grad, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingLookup(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& table_ids, const Symbol<DType>& dtype, const Symbol<DType>& embedding_dtype, int64_t line_size, int64_t embedding_size, const std::string& embedding_name, const std::string& embedding_tables, const std::string& state_initializer, int64_t seed);

Maybe<one::Tensor> OneEmbeddingFusedLookup(const std::shared_ptr<one::Tensor>& shadow, const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype, const std::string& embedding_name, int64_t line_size, int64_t embedding_size, bool is_full_cache, int32_t num_tables, const std::string& embedding_tables, const Optional<int64_t>& padding_idx, int64_t seed);

Maybe<void> OneEmbeddingFusedLookupGrad(const std::shared_ptr<one::Tensor>& ids, const std::shared_ptr<one::Tensor>& embedding_grad, const std::string& embedding_name, int64_t line_size, int64_t embedding_size);

Maybe<one::TensorTuple> OneEmbeddingUniqueKeyValuePair(const std::shared_ptr<one::Tensor>& keys, const Optional<one::Tensor>& values, int32_t num_tables, const std::string& embedding_name);

Maybe<void> OneEmbeddingEmbeddingPut(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::string& embedding_name, int64_t line_size);

Maybe<one::Tensor> OneEmbeddingSgdUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float momentum, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingAdamUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& bias_correction1, const Optional<one::Tensor>& bias_correction2, float learning_rate_val, double scale, float weight_decay, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, float epsilon, bool do_bias_correction, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingAdagradUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& train_step, int64_t train_step_val, float learning_rate_val, double scale, float weight_decay, float lr_decay, float epsilon, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);

Maybe<one::Tensor> OneEmbeddingFtrlUpdate(const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float lr_power, float lambda1, float lambda2, float beta, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);

Maybe<one::Tensor> EinSum(const std::string& equation, const TensorTuple& operands);

Maybe<one::Tensor> PixelShuffle(const std::shared_ptr<one::Tensor>& input, int64_t h_upscale_factor, int64_t w_upscale_factor);

Maybe<one::Tensor> IsNan(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> IsInf(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> IsFinite(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> RocAucScore(const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& pred);

Maybe<one::Tensor> PinMemory(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> FillTensor(const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& value);

Maybe<one::Tensor> Fill(const std::shared_ptr<one::Tensor>& in, const Scalar& value);

Maybe<one::Tensor> RnnTanhCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::Tensor> RnnReluCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::TensorTuple> LstmCell(const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::Tensor> GruCell(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::TensorTuple> FusedGruCell(const std::shared_ptr<one::Tensor>& igates, const std::shared_ptr<one::Tensor>& hgates, const std::shared_ptr<one::Tensor>& hx, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::TensorTuple> FusedGruCellGrad(const std::shared_ptr<one::Tensor>& grad_hy, const std::shared_ptr<one::Tensor>& workspace, bool has_bias, bool hx_needs_grad);

Maybe<one::TensorTuple> FusedLstmCell(const std::shared_ptr<one::Tensor>& igates, const std::shared_ptr<one::Tensor>& hgates, const std::shared_ptr<one::Tensor>& cx, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);

Maybe<one::TensorTuple> FusedLstmCellGrad(const std::shared_ptr<one::Tensor>& grad_hy, const std::shared_ptr<one::Tensor>& grad_cy, const std::shared_ptr<one::Tensor>& cx, const std::shared_ptr<one::Tensor>& cy, const std::shared_ptr<one::Tensor>& workspace, bool need_cx_grad, bool has_bias);

Maybe<one::TensorTuple> RnnTanhInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);

Maybe<one::TensorTuple> RnnTanhData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);

Maybe<one::TensorTuple> RnnReluInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);

Maybe<one::TensorTuple> RnnReluData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);

Maybe<one::TensorTuple> LstmInput(const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);

Maybe<one::TensorTuple> LstmData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);

Maybe<one::TensorTuple> GruInput(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);

Maybe<one::TensorTuple> GruData(const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);

Maybe<one::TensorTuple> PackPaddedSequence(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& lengths, bool batch_first);

Maybe<void> MultiTensorSgdUpdate(const TensorTuple& model, const TensorTuple& model_diff, double scale, float weight_decay, float learning_rate_val);

Maybe<void> MultiTensorAdamUpdate(const TensorTuple& model, const TensorTuple& model_diff, const TensorTuple& m, const TensorTuple& v, float learning_rate_val, float l2, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, bool do_bias_correction, double scale, float weight_decay, float epsilon);

Maybe<one::Tensor> GradAccRepeat(const std::shared_ptr<one::Tensor>& input, int32_t repeat_num);

Maybe<one::Tensor> GradAccCollect(const std::shared_ptr<one::Tensor>& input, int32_t collect_num);

Maybe<one::Tensor> GradAccPack(const std::shared_ptr<one::Tensor>& input, int32_t pack_num);

Maybe<one::Tensor> GradAccUnpack(const std::shared_ptr<one::Tensor>& input, int32_t unpack_num);

Maybe<one::Tensor> Trunc(const std::shared_ptr<one::Tensor>& input);

Maybe<one::Tensor> SiluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> MishGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SoftSignGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> GeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> HardSigmoidGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> HardSwishGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SoftplusGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double beta, double threshold);

Maybe<one::Tensor> EluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double alpha);

Maybe<one::Tensor> CeluGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx, double alpha);

Maybe<one::TensorTuple> BatchNormStats(const std::shared_ptr<one::Tensor>& input, int32_t axis, float eps);

Maybe<one::TensorTuple> BatchNormGatherStatsWithCounts(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const Optional<one::Tensor>& running_mean, const Optional<one::Tensor>& running_var, float momentum, float eps, const std::shared_ptr<one::Tensor>& counts);

Maybe<one::Tensor> BatchNormElemt(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& bias, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis, float eps);

Maybe<one::TensorTuple> BatchNormBackwardReduce(const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis);

Maybe<one::Tensor> BatchNormBackwardElemt(const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& sum_dy, const std::shared_ptr<one::Tensor>& sum_dy_xmu, const std::shared_ptr<one::Tensor>& count, int32_t axis);

Maybe<one::TensorTuple> AdaptiveMaxPool1D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);

Maybe<one::TensorTuple> AdaptiveMaxPool2D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);

Maybe<one::TensorTuple> AdaptiveMaxPool3D(const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);

Maybe<one::Tensor> AdaptiveMaxPoolNdGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& dy, int32_t ndims);

Maybe<one::Tensor> TanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SinhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> CoshGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> TanhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AcosGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AsinGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AtanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AsinhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AcoshGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> AtanhGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> ErfGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> ErfcGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> ExpGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Expm1GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> LogGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> LogSigmoidGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Log2GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Log10GradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> Log1pGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> ReciprocalGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> ReciprocalNoNanGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> RsqrtGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SqrtGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SquareGradGrad(const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> SigmoidGradGrad(const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dydx);

Maybe<one::Tensor> MaxPoolNdGradGrad(const std::shared_ptr<one::Tensor>& dydx, const std::shared_ptr<one::Tensor>& indices, int32_t ndims);

Maybe<one::Tensor> Exponential(const std::shared_ptr<one::Tensor>& x, float lambd, const Optional<one::Generator>& generator);

Maybe<one::Tensor> Multinomial(const std::shared_ptr<one::Tensor>& x, int32_t num_samples, bool replacement, const Optional<one::Generator>& generator);

Maybe<one::Tensor> DeformConv2d(const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const std::shared_ptr<one::Tensor>& mask, const Optional<one::Tensor>& bias, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask);

Maybe<one::TensorTuple> DeformConv2dInputGrad(const std::shared_ptr<one::Tensor>& output_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const Optional<one::Tensor>& mask, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask);

Maybe<one::Tensor> DeformConv2dParamGrad(const std::shared_ptr<one::Tensor>& output_grad, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const std::shared_ptr<one::Tensor>& mask, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask);

Maybe<one::Tensor> BinCount(const std::shared_ptr<one::Tensor>& input, const Optional<one::Tensor>& weights, const Optional<int64_t>& minlength);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_GENERATED_FUNCTIONAL_API_H_